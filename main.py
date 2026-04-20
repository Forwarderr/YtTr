import os
import re
import logging
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, field_validator

from youtube_transcript_api import (
    YouTubeTranscriptApi,
    NoTranscriptFound,
    TranscriptsDisabled,
    VideoUnavailable,
    CouldNotRetrieveTranscript,
)
from youtube_transcript_api.proxies import WebshareProxyConfig

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger("transcript_api")

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="YouTube Transcript API",
    description="Fetch timestamped transcripts from YouTube videos.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],        # Tighten in production
    allow_methods=["GET"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Response schemas
# ---------------------------------------------------------------------------
class TranscriptLine(BaseModel):
    timestamp: str          # "[MM:SS]"
    start_seconds: float
    text: str


class TranscriptResponse(BaseModel):
    video_id: str
    language: str
    language_code: str
    is_generated: bool
    line_count: int
    lines: list[TranscriptLine]


class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
_YT_URL_RE = re.compile(
    r"(?:youtube\.com/(?:[^/]+/.+/|(?:v|e(?:mbed)?)/|.*[?&]v=)|youtu\.be/)"
    r"([A-Za-z0-9_-]{11})"
)

def extract_video_id(url: str) -> Optional[str]:
    """Return the 11-char video ID from any YouTube URL, or None."""
    url = url.strip()
    # Accept a bare video ID too (exactly 11 URL-safe chars)
    if re.fullmatch(r"[A-Za-z0-9_-]{11}", url):
        return url
    match = _YT_URL_RE.search(url)
    return match.group(1) if match else None


def format_time(seconds: float) -> str:
    """Convert raw seconds → '[MM:SS]'."""
    total = int(seconds)
    return f"[{total // 60:02d}:{total % 60:02d}]"


def fetch_transcript(video_id: str, language: Optional[str]) -> TranscriptResponse:
    """
    Core logic: list transcripts, optionally filter by language, fetch & format.
    Raises HTTPException with appropriate status codes on failure.
    """
    try:
        proxy_config = WebshareProxyConfig(
            proxy_username=os.getenv("PROXY_USER"),
            proxy_password=os.getenv("PROXY_PASS"),
        )
        api = YouTubeTranscriptApi(proxy_config=proxy_config)
        transcript_list = api.list(video_id)
    except VideoUnavailable:
        logger.warning("Video unavailable: %s", video_id)
        raise HTTPException(status_code=404, detail=f"Video '{video_id}' is unavailable or does not exist.")
    except TranscriptsDisabled:
        logger.warning("Transcripts disabled: %s", video_id)
        raise HTTPException(status_code=403, detail=f"Transcripts are disabled for video '{video_id}'.")
    except CouldNotRetrieveTranscript as exc:
        logger.error("Could not retrieve transcript for %s: %s", video_id, exc)
        raise HTTPException(status_code=502, detail="Could not retrieve transcript from YouTube.")
    except Exception as exc:
        logger.exception("Unexpected error listing transcripts for %s", video_id)
        raise HTTPException(status_code=500, detail=f"Unexpected error: {exc}")

    # Pick transcript: prefer requested language, else first available
    try:
        if language:
            transcript = transcript_list.find_transcript([language])
        else:
            transcript = next(iter(transcript_list))
    except NoTranscriptFound:
        available = [t.language_code for t in transcript_list]
        raise HTTPException(
            status_code=404,
            detail=f"No transcript found for language '{language}'. Available: {available}",
        )
    except StopIteration:
        raise HTTPException(status_code=404, detail="No transcripts found for this video.")

    # Fetch transcript data
    try:
        raw_data = transcript.fetch().to_raw_data()
    except Exception as exc:
        logger.exception("Error fetching transcript data for %s", video_id)
        raise HTTPException(status_code=502, detail=f"Failed to fetch transcript data: {exc}")

    if not raw_data:
        raise HTTPException(status_code=204, detail="Transcript is empty.")

    lines = [
        TranscriptLine(
            timestamp=format_time(item["start"]),
            start_seconds=round(item["start"], 3),
            text=item["text"],
        )
        for item in raw_data
        if item.get("text", "").strip()   # skip blank lines
    ]

    return TranscriptResponse(
        video_id=video_id,
        language=transcript.language,
        language_code=transcript.language_code,
        is_generated=transcript.is_generated,
        line_count=len(lines),
        lines=lines,
    )


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/", tags=["Health"])
def health_check():
    """Basic liveness check."""
    return {"status": "ok", "service": "YouTube Transcript API"}


@app.get(
    "/transcript",
    response_model=TranscriptResponse,
    responses={
        400: {"model": ErrorResponse},
        403: {"model": ErrorResponse},
        404: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
        502: {"model": ErrorResponse},
    },
    tags=["Transcript"],
    summary="Fetch a timestamped transcript",
)
def get_transcript(
    url: str = Query(..., description="Full YouTube URL or bare 11-char video ID"),
    language: Optional[str] = Query(
        None,
        description="BCP-47 language code (e.g. 'en', 'es'). Defaults to first available.",
        min_length=2,
        max_length=10,
    ),
):
    """
    Returns a structured, timestamped transcript for the given YouTube video.

    - **url**: Full YouTube URL *or* a bare 11-character video ID.
    - **language**: Optional BCP-47 language code. Falls back to the first available transcript.
    """
    video_id = extract_video_id(url)
    if not video_id:
        raise HTTPException(
            status_code=400,
            detail="Could not parse a valid YouTube video ID from the provided URL.",
        )

    logger.info("Transcript request — video_id=%s language=%s", video_id, language)
    return fetch_transcript(video_id, language)


# ---------------------------------------------------------------------------
# Global exception handler (catch-all safety net)
# ---------------------------------------------------------------------------
@app.exception_handler(Exception)
async def unhandled_exception_handler(request, exc):
    logger.exception("Unhandled exception on %s", request.url)
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "detail": str(exc)},
    )


# ---------------------------------------------------------------------------
# Entry point (dev)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
