from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # --- GCP / Vertex AI ---
    gcp_project_id: str = ""            # optional — ADC picks up project from gcloud config if empty
    google_api_key: str = ""            # not needed for Vertex AI (uses gcloud credentials)
    anthropic_api_key: str = ""         # kept for backward-compat; no longer used by LLMClient

    # --- Asset APIs ---
    pexels_api_key: str
    pixabay_api_key: str

    # Optional — only needed for Stage 9 publish
    youtube_client_id: str = ""
    youtube_client_secret: str = ""
    youtube_refresh_token: str = ""

    # --- Filesystem roots ---
    projects_root: Path = Field(default=Path("projects"))
    style_pack_root: Path = Field(default=Path("style_pack"))
    cache_dir: Path = Field(default=Path(".cache/diskcache"))

    # --- ElevenLabs TTS ---
    elevenlabs_api_key: str = ""       # set to use ElevenLabs instead of Kokoro
    elevenlabs_voice_id: str = ""      # optional — resolved by name search if empty
    elevenlabs_voice_name: str = "Ryan Kurk"  # fallback name search

    # --- Pipeline overrides (mirror constants.py defaults, env-settable) ---
    tts_backend: str = "kokoro"        # "kokoro" | "elevenlabs"
    tts_device: str = "cpu"
    whisperx_model: str = "large-v3"
    clip_model: str = "ViT-B-32"
    max_parallel_assets: int = 8
    ffmpeg_video_encoder: str = "h264_videotoolbox"
    render_max_workers: int = 4

    @field_validator("projects_root", "style_pack_root", "cache_dir", mode="before")
    @classmethod
    def _to_path(cls, v: str | Path) -> Path:
        return Path(v)

    @property
    def voice_ref_path(self) -> Path:
        return self.style_pack_root / "voice_ref.wav"

    @property
    def fonts_dir(self) -> Path:
        return self.style_pack_root / "fonts"

    @property
    def music_dir(self) -> Path:
        return self.style_pack_root / "music"

    @property
    def sfx_dir(self) -> Path:
        return self.style_pack_root / "sfx"

    @property
    def colors_file(self) -> Path:
        return self.style_pack_root / "colors.json"

    def youtube_configured(self) -> bool:
        return bool(self.youtube_client_id and self.youtube_client_secret)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
