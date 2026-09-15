"""
Song Curator
============

Quickly listen to the most active portion of audio files and classify them
as liked or disliked.

Features
--------
- Drag and drop audio files and folders onto a large drop area.
- Recursively scans dropped folders, ignores non-audio files and duplicates.
- Uses FFprobe for duration and FFmpeg to find the most active X-second section.
- Large LIKE / DISLIKE buttons plus keyboard shortcuts.
- Separate, always-in-sync lists for All / Remaining / Liked / Disliked songs.
- Optional auto-dislike: if the preview ends without a LIKE, the song is
  disliked automatically and the next one starts.
- Session is saved continuously, so a crash or close can be resumed with all
  liked, disliked, skipped and remaining songs intact.
- Copy or move liked files to an output directory with SHA-256 duplicate
  detection (identical files skipped, different files overwritten).
- Delete every disliked file from disk, to the recycle bin when send2trash is
  installed and permanently otherwise.
- FFmpeg / playback failures skip the song instead of crashing.
- Settings are saved to JSON and restored on startup.
- Logging to console and song_curator.log.

Requirements
------------
    pip install pygame tkinterdnd2 send2trash

send2trash is optional. Without it, deleting disliked files is permanent.

FFmpeg
------
FFmpeg and FFprobe must be on PATH or specified in the Settings section.

Python
------
Python 3.10+
"""

from __future__ import annotations

import array
import functools
import hashlib
import json
import logging
import os
import shutil
import subprocess
import threading
import time

from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional, Sequence

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import pygame

import tkinterdnd2
from tkinterdnd2 import DND_FILES
from send2trash import send2trash


# ============================================================================
# Application constants
# ============================================================================

APPLICATION_NAME = "Song Curator"

SETTINGS_FILE_NAME = "song_curator_settings.json"
SESSION_FILE_NAME = "song_curator_session.json"
LOG_FILE_NAME = "song_curator.log"

DEFAULT_PREVIEW_SECONDS = 30.0

AUDIO_EXTENSIONS = {
    ".mp3", ".m4a", ".flac", ".wav", ".ogg", ".oga", ".opus", ".aac",
    ".wma", ".aiff", ".aif", ".mp2", ".webm", ".mka",
}

# Audio is converted to mono 16-bit PCM at this sample rate for analysis.
ANALYSIS_SAMPLE_RATE = 16000

# Activity is calculated in blocks of this length.
ACTIVITY_BLOCK_SECONDS = 0.25

HASH_BUFFER_SIZE = 1024 * 1024
COPY_BUFFER_SIZE = 1024 * 1024

# How often the GUI checks whether a preview should be stopped.
PLAYBACK_CHECK_INTERVAL_MS = 250

# How long to wait after a change before writing the session file.
SESSION_SAVE_DELAY_MS = 750

SESSION_VERSION = 1

# Retries for files that are still briefly locked when a delete is attempted.
DELETE_ATTEMPTS = 3
DELETE_RETRY_SECONDS = 0.3

# Classification states.
STATUS_PENDING = "pending"
STATUS_LIKED = "liked"
STATUS_DISLIKED = "disliked"
STATUS_SKIPPED = "skipped"

VALID_STATUSES = {STATUS_PENDING, STATUS_LIKED, STATUS_DISLIKED, STATUS_SKIPPED}


# ============================================================================
# Logging
# ============================================================================

def configure_logging() -> logging.Logger:
    """Configure application logging."""

    logger = logging.getLogger(APPLICATION_NAME)
    logger.setLevel(logging.INFO)

    if logger.handlers:
        return logger

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    file_handler = logging.FileHandler(LOG_FILE_NAME, encoding="utf-8")
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


LOGGER = configure_logging()


# ============================================================================
# Helpers
# ============================================================================

@functools.lru_cache(maxsize=100_000)
def normalize_path(text: str) -> str:
    """Normalize a path string for duplicate detection."""

    try:
        return os.path.normcase(str(Path(text).resolve()))
    except OSError:
        return os.path.normcase(text)


# ============================================================================
# Data classes
# ============================================================================

@dataclass(frozen=True)
class Song:
    """Represents a song."""

    path: Path

    @property
    def name(self) -> str:
        """Return the filename."""

        return self.path.name

    @property
    def key(self) -> str:
        """Return the normalized identity of this song."""

        return normalize_path(str(self.path))


@dataclass
class AnalysisResult:
    """Result returned by FFmpeg audio analysis."""

    duration_seconds: float
    active_start_seconds: float


@dataclass
class ExportResult:
    """Results from an export operation."""

    copied: int = 0
    moved: int = 0
    skipped: int = 0
    errors: int = 0
    error_messages: list[str] = field(default_factory=list)
    moved_pairs: list[tuple[Path, Path]] = field(default_factory=list)


@dataclass
class DeleteResult:
    """Results from a delete operation."""

    deleted: int = 0
    missing: int = 0
    errors: int = 0
    used_recycle_bin: bool = False
    error_messages: list[str] = field(default_factory=list)

    # Songs whose file is gone, so they can be dropped from the lists.
    removed_paths: list[Path] = field(default_factory=list)


# ============================================================================
# Settings
# ============================================================================

class SettingsManager:
    """Loads and saves application settings as JSON."""

    def __init__(self, settings_path: Path) -> None:
        self.settings_path = settings_path

    def load(self) -> dict[str, str]:
        """Load settings from disk."""

        if not self.settings_path.exists():
            return {}

        try:
            with self.settings_path.open("r", encoding="utf-8") as file:
                data = json.load(file)

            if not isinstance(data, dict):
                LOGGER.warning("Settings file did not contain a JSON object.")
                return {}

            result = {
                str(key): str(value)
                for key, value in data.items()
                if isinstance(value, (str, int, float, bool))
            }

            LOGGER.info("Loaded settings from %s.", self.settings_path)
            return result

        except Exception:
            LOGGER.exception("Could not load settings file: %s", self.settings_path)
            return {}

    def save(self, settings: dict[str, str]) -> None:
        """Save settings to disk."""

        try:
            with self.settings_path.open("w", encoding="utf-8") as file:
                json.dump(settings, file, indent=4)

            LOGGER.info("Saved settings to %s.", self.settings_path)

        except Exception:
            LOGGER.exception("Could not save settings file: %s", self.settings_path)


# ============================================================================
# Session persistence
# ============================================================================

class SessionManager:
    """
    Saves and restores the review session.

    The file is written atomically through a temporary file so a crash during
    the write cannot corrupt the previous session.
    """

    def __init__(self, session_path: Path) -> None:
        self.session_path = session_path

    def load(self) -> list[tuple[Path, str]]:
        """Load the saved session as (path, status) pairs."""

        if not self.session_path.exists():
            return []

        try:
            with self.session_path.open("r", encoding="utf-8") as file:
                data = json.load(file)

            if not isinstance(data, dict):
                LOGGER.warning("Session file did not contain a JSON object.")
                return []

            records = data.get("songs", [])

            if not isinstance(records, list):
                LOGGER.warning("Session file did not contain a song list.")
                return []

            result: list[tuple[Path, str]] = []

            for record in records:
                if not isinstance(record, dict):
                    continue

                raw_path = record.get("path")
                status = str(record.get("status", STATUS_PENDING)).lower()

                if not isinstance(raw_path, str) or not raw_path:
                    continue

                if status not in VALID_STATUSES:
                    status = STATUS_PENDING

                result.append((Path(raw_path), status))

            LOGGER.info("Loaded %d song(s) from %s.", len(result), self.session_path)
            return result

        except Exception:
            LOGGER.exception("Could not load session file: %s", self.session_path)
            return []

    def save(self, records: Sequence[tuple[Path, str]]) -> None:
        """Save the session to disk atomically."""

        payload = {
            "version": SESSION_VERSION,
            "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "songs": [{"path": str(path), "status": status} for path, status in records],
        }

        temporary_path = self.session_path.with_suffix(".tmp")

        try:
            with temporary_path.open("w", encoding="utf-8") as file:
                json.dump(payload, file, indent=1)
                file.flush()
                os.fsync(file.fileno())

            os.replace(temporary_path, self.session_path)

        except Exception:
            LOGGER.exception("Could not save session file: %s", self.session_path)

    def clear(self) -> None:
        """Delete the saved session."""

        try:
            self.session_path.unlink(missing_ok=True)
            LOGGER.info("Cleared session file: %s", self.session_path)
        except Exception:
            LOGGER.exception("Could not clear session file: %s", self.session_path)


# ============================================================================
# Audio file discovery
# ============================================================================

class AudioFileManager:
    """Discovers supported audio files."""

    def __init__(self, extensions: set[str]) -> None:
        self.extensions = {extension.lower() for extension in extensions}

    def is_audio_file(self, path: Path) -> bool:
        """Return whether a path is a supported audio file."""

        return path.is_file() and path.suffix.lower() in self.extensions

    def discover(self, paths: Sequence[Path]) -> list[Path]:
        """Discover audio files. Files are accepted directly, folders are scanned."""

        result: list[Path] = []

        for original_path in paths:
            try:
                path = original_path.expanduser().resolve()
            except OSError:
                LOGGER.exception("Could not resolve path: %s", original_path)
                continue

            if path.is_file():
                if self.is_audio_file(path):
                    result.append(path)
                continue

            if not path.is_dir():
                LOGGER.warning("Path does not exist or is not a directory: %s", path)
                continue

            try:
                for child in path.rglob("*"):
                    try:
                        if self.is_audio_file(child):
                            result.append(child.resolve())
                    except OSError:
                        LOGGER.warning("Could not inspect: %s", child)

            except OSError:
                LOGGER.exception("Could not scan directory: %s", path)

        return result


# ============================================================================
# FFmpeg
# ============================================================================

class FFmpegManager:
    """
    Handles FFmpeg and FFprobe operations.

    FFmpeg decodes audio into raw PCM for activity analysis.
    FFprobe provides accurate duration information.
    """

    def __init__(self, ffmpeg_path: str = "ffmpeg", ffprobe_path: str = "ffprobe") -> None:
        self.ffmpeg_path = ffmpeg_path
        self.ffprobe_path = ffprobe_path

    def set_paths(self, ffmpeg_path: str, ffprobe_path: str) -> None:
        """Update executable paths."""

        self.ffmpeg_path = ffmpeg_path.strip() or "ffmpeg"
        self.ffprobe_path = ffprobe_path.strip() or "ffprobe"

    @staticmethod
    def _subprocess_flags() -> int:
        """Return flags that hide console windows on Windows."""

        return getattr(subprocess, "CREATE_NO_WINDOW", 0)

    def get_duration(self, path: Path) -> float:
        """Use FFprobe to obtain the audio duration."""

        command = [
            self.ffprobe_path, "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(path),
        ]

        LOGGER.info("Running FFprobe for duration: %s", path)

        completed = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
            creationflags=self._subprocess_flags(),
        )

        if completed.returncode != 0:
            raise RuntimeError(
                f"FFprobe failed with exit code {completed.returncode}: "
                f"{completed.stderr.strip()}"
            )

        output = completed.stdout.strip()

        try:
            duration = float(output)
        except ValueError as exc:
            raise RuntimeError(f"FFprobe returned an invalid duration: {output!r}") from exc

        if duration <= 0:
            raise RuntimeError("FFprobe reported a duration of zero.")

        return duration

    def find_most_active_section(
        self,
        path: Path,
        preview_seconds: float,
        duration_seconds: float,
        progress_callback: Optional[Callable[[float], None]] = None,
    ) -> float:
        """
        Find the most active contiguous section of the requested length.

        Audio is decoded to mono 16-bit PCM, split into small blocks, and the
        RMS energy of each block is summed over a sliding window. The window
        with the highest total energy wins.

        Returns:
            Start time of the most active section.
        """

        if duration_seconds <= preview_seconds:
            return 0.0

        block_seconds = ACTIVITY_BLOCK_SECONDS

        command = [
            self.ffmpeg_path, "-hide_banner", "-loglevel", "error",
            "-i", str(path),
            "-vn", "-sn", "-dn",
            "-ac", "1",
            "-ar", str(ANALYSIS_SAMPLE_RATE),
            "-f", "s16le", "pipe:1",
        ]

        LOGGER.info("Analyzing activity with FFmpeg: %s", path)

        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            creationflags=self._subprocess_flags(),
        )

        if process.stdout is None:
            process.kill()
            raise RuntimeError("Could not open FFmpeg stdout.")

        bytes_per_sample = 2
        bytes_per_block = int(ANALYSIS_SAMPLE_RATE * block_seconds * bytes_per_sample)

        if bytes_per_block <= 0:
            process.kill()
            raise RuntimeError("Invalid FFmpeg analysis block size.")

        window_blocks = max(1, int(round(preview_seconds / block_seconds)))

        energy_window: deque[tuple[int, float]] = deque()
        energy_sum = 0.0

        best_energy = -1.0
        best_block_index = 0

        block_index = 0
        bytes_processed = 0
        last_progress = 0.0

        try:
            while True:
                raw_block = process.stdout.read(bytes_per_block)

                if not raw_block:
                    break

                usable = (len(raw_block) // bytes_per_sample) * bytes_per_sample
                raw_block = raw_block[:usable]

                if not raw_block:
                    continue

                energy = self._calculate_rms(raw_block)

                energy_window.append((block_index, energy))
                energy_sum += energy

                if len(energy_window) > window_blocks:
                    _, old_energy = energy_window.popleft()
                    energy_sum -= old_energy

                if len(energy_window) == window_blocks and energy_sum > best_energy:
                    best_energy = energy_sum
                    best_block_index = energy_window[0][0]

                block_index += 1
                bytes_processed += len(raw_block)

                if progress_callback is not None:
                    estimated_seconds = bytes_processed / (ANALYSIS_SAMPLE_RATE * bytes_per_sample)

                    # Report at most a few times per second of audio.
                    if estimated_seconds - last_progress >= 1.0:
                        last_progress = estimated_seconds
                        progress_callback(estimated_seconds)

        finally:
            try:
                process.stdout.close()
            except Exception:
                pass

        stderr_data = b""

        if process.stderr is not None:
            try:
                stderr_data = process.stderr.read()
            except Exception:
                stderr_data = b""

            try:
                process.stderr.close()
            except Exception:
                pass

        return_code = process.wait()

        if return_code != 0:
            stderr_text = stderr_data.decode("utf-8", errors="replace").strip()
            raise RuntimeError(f"FFmpeg failed with exit code {return_code}: {stderr_text}")

        if best_energy < 0:
            raise RuntimeError("FFmpeg did not provide enough audio data for analysis.")

        active_start = best_block_index * block_seconds

        # Never let the preview run past the end of the song.
        max_start = max(0.0, duration_seconds - preview_seconds)

        return min(active_start, max_start)

    @staticmethod
    def _calculate_rms(pcm_data: bytes) -> float:
        """Calculate normalized RMS energy from signed 16-bit little-endian PCM."""

        samples = array.array("h")
        samples.frombytes(pcm_data)

        if not samples:
            return 0.0

        # array uses native byte order, so swap on big-endian machines.
        if array.array("h", b"\x01\x00")[0] != 1:
            samples.byteswap()

        total_square = sum(float(sample) * float(sample) for sample in samples)

        return (total_square / len(samples)) ** 0.5 / 32768.0


# ============================================================================
# Audio player
# ============================================================================

class AudioPlayer:
    """Handles audio playback using pygame."""

    def __init__(self) -> None:
        self.initialized = False

        self.play_started_at = 0.0
        self.play_duration_seconds = 0.0
        self.elapsed_before_pause = 0.0
        self.is_paused = False

        self._initialize()

    def _initialize(self) -> None:
        """Initialize the pygame mixer."""

        try:
            pygame.mixer.init()
            self.initialized = True
            LOGGER.info("Pygame audio initialized.")
        except Exception:
            LOGGER.exception("Could not initialize pygame audio.")

    def play(self, path: Path, start_seconds: float, duration_seconds: float) -> None:
        """Start or restart a preview from the specified position."""

        if not self.initialized:
            raise RuntimeError("Pygame audio is not initialized.")

        self.stop()

        LOGGER.info(
            "Playing %s from %.2f seconds for %.2f seconds.",
            path, start_seconds, duration_seconds,
        )

        pygame.mixer.music.load(str(path))

        try:
            pygame.mixer.music.play(loops=0, start=max(0.0, start_seconds))

        except (TypeError, pygame.error):
            # Some pygame/SDL combinations cannot use the start parameter for
            # certain codecs, so play normally and then seek.
            pygame.mixer.music.play()

            try:
                pygame.mixer.music.set_pos(start_seconds)
            except Exception as exc:
                LOGGER.warning("Could not seek to %.2f seconds for %s: %s", start_seconds, path, exc)

        self.play_started_at = time.monotonic()
        self.play_duration_seconds = max(0.0, duration_seconds)
        self.elapsed_before_pause = 0.0
        self.is_paused = False

    def pause(self) -> None:
        """Pause the current preview."""

        if not self.initialized or not self.is_playing():
            return

        try:
            elapsed = time.monotonic() - self.play_started_at
            self.elapsed_before_pause += max(0.0, elapsed)

            pygame.mixer.music.pause()
            self.is_paused = True

            LOGGER.info("Playback paused.")

        except Exception:
            LOGGER.exception("Could not pause playback.")

    def resume(self) -> None:
        """Resume paused playback."""

        if not self.initialized or not self.is_paused:
            return

        try:
            pygame.mixer.music.unpause()
            self.play_started_at = time.monotonic()
            self.is_paused = False

            LOGGER.info("Playback resumed.")

        except Exception:
            LOGGER.exception("Could not resume playback.")

    def stop(self) -> None:
        """Stop playback and reset playback state."""

        if not self.initialized:
            return

        try:
            pygame.mixer.music.stop()
        except Exception:
            LOGGER.exception("Could not stop playback.")

        self.play_started_at = 0.0
        self.play_duration_seconds = 0.0
        self.elapsed_before_pause = 0.0
        self.is_paused = False

    def release(self) -> None:
        """
        Stop playback and release the loaded file.

        Pygame keeps a handle on the most recently loaded track even after it
        stops, which prevents the file from being deleted or moved on Windows.
        """

        self.stop()

        if not self.initialized:
            return

        try:
            pygame.mixer.music.unload()
            LOGGER.info("Released the loaded audio file.")
            return

        except Exception:
            LOGGER.info("Mixer has no unload(). Restarting it to release the file.")

        # Older pygame builds have no unload(), so cycle the mixer instead.
        try:
            pygame.mixer.quit()
            pygame.mixer.init()
        except Exception:
            LOGGER.exception("Could not restart the mixer to release the file.")
            self.initialized = False

    def is_playing(self) -> bool:
        """Return whether pygame currently reports active playback."""

        if not self.initialized or self.is_paused:
            return False

        try:
            return bool(pygame.mixer.music.get_busy())
        except Exception:
            return False

    def is_paused_state(self) -> bool:
        """Return whether playback is currently paused."""

        return self.is_paused

    def preview_expired(self) -> bool:
        """Return whether the configured preview duration elapsed."""

        if self.play_started_at <= 0 or self.is_paused:
            return False

        elapsed = time.monotonic() - self.play_started_at + self.elapsed_before_pause

        return elapsed >= self.play_duration_seconds

    def shutdown(self) -> None:
        """Shut down the audio system."""

        try:
            self.stop()
            pygame.mixer.quit()
        except Exception:
            LOGGER.exception("Could not shut down pygame.")


# ============================================================================
# SHA-256
# ============================================================================

class HashManager:
    """Calculates SHA-256 hashes."""

    @staticmethod
    def sha256(path: Path) -> str:
        """Calculate a SHA-256 hash incrementally."""

        digest = hashlib.sha256()

        with path.open("rb") as file:
            while True:
                chunk = file.read(HASH_BUFFER_SIZE)

                if not chunk:
                    break

                digest.update(chunk)

        return digest.hexdigest()

    def are_identical(self, source: Path, destination: Path) -> bool:
        """Determine whether two files have identical contents."""

        if source.stat().st_size != destination.stat().st_size:
            return False

        return self.sha256(source) == self.sha256(destination)


# ============================================================================
# Output manager
# ============================================================================

class OutputManager:
    """Copies or moves liked songs."""

    def __init__(self, hash_manager: HashManager) -> None:
        self.hash_manager = hash_manager

    def export(
        self,
        songs: Sequence[Song],
        output_directory: Path,
        move_files: bool,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
    ) -> ExportResult:
        """Export songs to the output directory."""

        result = ExportResult()

        output_directory.mkdir(parents=True, exist_ok=True)

        total = len(songs)

        for index, song in enumerate(songs, start=1):
            source = song.path
            destination = output_directory / source.name

            if progress_callback is not None:
                progress_callback(index, total, source.name)

            try:
                if not source.exists():
                    raise FileNotFoundError(f"Source file no longer exists: {source}")

                if destination.exists():
                    LOGGER.info("Destination exists. Checking SHA-256: %s", destination)

                    if self.hash_manager.are_identical(source, destination):
                        result.skipped += 1
                        LOGGER.info("Skipping identical file: %s", destination)
                        continue

                    LOGGER.info("Destination differs and will be overwritten: %s", destination)

                if move_files:
                    self._move(source, destination)
                    result.moved += 1
                    result.moved_pairs.append((source, destination))
                else:
                    self._copy(source, destination)
                    result.copied += 1

            except Exception as exc:
                result.errors += 1
                result.error_messages.append(f"{source.name}: {exc}")
                LOGGER.exception("Could not export %s.", source)

        return result

    @staticmethod
    def delete(
        songs: Sequence[Song],
        use_recycle_bin: bool,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
    ) -> DeleteResult:
        """Delete song files from disk."""

        result = DeleteResult(used_recycle_bin=use_recycle_bin)

        total = len(songs)

        for index, song in enumerate(songs, start=1):
            path = song.path

            if progress_callback is not None:
                progress_callback(index, total, path.name)

            try:
                if not path.exists():
                    result.missing += 1
                    result.removed_paths.append(path)
                    LOGGER.info("File was already gone: %s", path)
                    continue

                # A player or scanner may still hold the file for a moment, so
                # give it a couple of short retries before giving up.
                for attempt in range(DELETE_ATTEMPTS):
                    try:
                        if use_recycle_bin:
                            send2trash(str(path))
                        else:
                            path.unlink()
                        break

                    except OSError:
                        if attempt == DELETE_ATTEMPTS - 1:
                            raise

                        LOGGER.warning("File is busy, retrying: %s", path)
                        time.sleep(DELETE_RETRY_SECONDS)

                result.deleted += 1
                result.removed_paths.append(path)

                LOGGER.info("Deleted %s (recycle bin: %s).", path, use_recycle_bin)

            except Exception as exc:
                result.errors += 1
                result.error_messages.append(f"{path.name}: {exc}")
                LOGGER.exception("Could not delete %s.", path)

        return result

    @staticmethod
    def _copy(source: Path, destination: Path) -> None:
        """Copy a file while preserving metadata."""

        with source.open("rb") as source_file, destination.open("wb") as destination_file:
            while True:
                chunk = source_file.read(COPY_BUFFER_SIZE)

                if not chunk:
                    break

                destination_file.write(chunk)

        shutil.copystat(source, destination)

    @staticmethod
    def _move(source: Path, destination: Path) -> None:
        """Move a file."""

        if destination.exists():
            destination.unlink()

        shutil.move(str(source), str(destination))


# ============================================================================
# Song queue
# ============================================================================

class SongQueue:
    """
    Maintains every song and its classification state.

    A single ordered list of songs plus a status map is the source of truth,
    so the All / Remaining / Liked / Disliked views can never drift apart.
    """

    def __init__(self) -> None:
        self.songs: list[Song] = []
        self._status: dict[str, str] = {}

    # -- Queries ------------------------------------------------------------

    @property
    def pending(self) -> list[Song]:
        """Return songs still waiting to be classified."""

        return self._with_status(STATUS_PENDING)

    @property
    def liked(self) -> list[Song]:
        """Return liked songs."""

        return self._with_status(STATUS_LIKED)

    @property
    def disliked(self) -> list[Song]:
        """Return disliked songs."""

        return self._with_status(STATUS_DISLIKED)

    @property
    def skipped(self) -> list[Song]:
        """Return songs that could not be analyzed or played."""

        return self._with_status(STATUS_SKIPPED)

    @property
    def current_song(self) -> Optional[Song]:
        """Return the first pending song."""

        for song in self.songs:
            if self._status.get(song.key) == STATUS_PENDING:
                return song

        return None

    @property
    def remaining_count(self) -> int:
        """Return the number of songs still waiting."""

        return sum(1 for song in self.songs if self._status.get(song.key) == STATUS_PENDING)

    def status_of(self, song: Song) -> str:
        """Return the status of a song."""

        return self._status.get(song.key, STATUS_PENDING)

    def records(self) -> list[tuple[Path, str]]:
        """Return (path, status) pairs for persistence."""

        return [(song.path, self.status_of(song)) for song in self.songs]

    def _with_status(self, status: str) -> list[Song]:
        return [song for song in self.songs if self._status.get(song.key) == status]

    # -- Mutations ----------------------------------------------------------

    def add(self, paths: Sequence[Path], status: str = STATUS_PENDING) -> int:
        """Add songs while ignoring duplicates."""

        added = 0

        for path in paths:
            song = Song(path=path)

            if song.key in self._status:
                continue

            self.songs.append(song)
            self._status[song.key] = status
            added += 1

        return added

    def set_status(self, song: Song, status: str) -> None:
        """Set the status of a song."""

        if song.key in self._status:
            self._status[song.key] = status

    def classify_current(self, liked: bool) -> Optional[Song]:
        """Classify the current song and return it."""

        song = self.current_song

        if song is None:
            return None

        self._status[song.key] = STATUS_LIKED if liked else STATUS_DISLIKED

        return song

    def requeue(self, song: Song) -> None:
        """Send a song back to the queue so it is reviewed next."""

        if song.key not in self._status:
            return

        self._status[song.key] = STATUS_PENDING

        try:
            self.songs.remove(song)
        except ValueError:
            return

        insert_at = len(self.songs)

        for index, other in enumerate(self.songs):
            if self._status.get(other.key) == STATUS_PENDING:
                insert_at = index
                break

        self.songs.insert(insert_at, song)

    def relocate(self, old_path: Path, new_path: Path) -> None:
        """Update a song's path after it was moved on disk."""

        old_song = Song(path=old_path)
        status = self._status.pop(old_song.key, None)

        if status is None:
            return

        new_song = Song(path=new_path)

        try:
            index = self.songs.index(old_song)
            self.songs[index] = new_song
        except ValueError:
            self.songs.append(new_song)

        self._status[new_song.key] = status

    def remove(self, songs: Sequence[Song]) -> None:
        """Remove songs entirely."""

        keys = {song.key for song in songs}

        self.songs = [song for song in self.songs if song.key not in keys]

        for key in keys:
            self._status.pop(key, None)

    def clear_pending(self) -> None:
        """Remove pending songs."""

        self.remove(self.pending)

    def clear_liked(self) -> None:
        """Remove liked songs."""

        self.remove(self.liked)

    def clear_disliked(self) -> None:
        """Remove disliked songs."""

        self.remove(self.disliked)

    def clear_everything(self) -> None:
        """Remove every song."""

        self.songs.clear()
        self._status.clear()


# ============================================================================
# Main GUI
# ============================================================================

class SongCuratorApp:
    """Main Song Curator GUI."""

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title(APPLICATION_NAME)
        self.root.geometry("1150x900")
        self.root.minsize(950, 750)

        base_directory = Path(__file__).resolve().parent

        self.settings_manager = SettingsManager(base_directory / SETTINGS_FILE_NAME)
        self.session_manager = SessionManager(base_directory / SESSION_FILE_NAME)

        self.audio_file_manager = AudioFileManager(AUDIO_EXTENSIONS)
        self.ffmpeg_manager = FFmpegManager()
        self.audio_player = AudioPlayer()
        self.hash_manager = HashManager()
        self.output_manager = OutputManager(self.hash_manager)
        self.song_queue = SongQueue()

        # Runtime state.
        self.current_analysis: Optional[AnalysisResult] = None
        self.displayed_song: Optional[Song] = None
        self.is_analyzing = False
        self.is_exporting = False
        self.is_deleting = False
        self.current_song_token = 0
        self._session_save_job: Optional[str] = None

        # True once the user starts continuous playback.
        self.auto_play = False

        # Variables.
        self.preview_seconds_var = tk.StringVar()
        self.output_directory_var = tk.StringVar()
        self.operation_var = tk.StringVar()
        self.ffmpeg_path_var = tk.StringVar()
        self.ffprobe_path_var = tk.StringVar()
        self.auto_dislike_var = tk.BooleanVar(value=False)

        self.song_name_var = tk.StringVar(value="No song selected")
        self.song_path_var = tk.StringVar(value="")
        self.position_var = tk.StringVar(value="")
        self.status_var = tk.StringVar(value="Add audio files to begin.")
        self.counts_var = tk.StringVar(value="")

        self._load_settings()
        self._build_ui()
        self._configure_drag_and_drop()
        self._bind_shortcuts()

        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        self._restore_session()

        self.root.after(PLAYBACK_CHECK_INTERVAL_MS, self._playback_timer)

    # ------------------------------------------------------------------
    # Settings
    # ------------------------------------------------------------------

    def _load_settings(self) -> None:
        """Load persisted settings."""

        settings = self.settings_manager.load()

        self.preview_seconds_var.set(settings.get("preview_seconds", str(DEFAULT_PREVIEW_SECONDS)))
        self.output_directory_var.set(settings.get("output_directory", ""))
        self.operation_var.set(settings.get("operation", "Copy"))
        self.ffmpeg_path_var.set(settings.get("ffmpeg_path", "ffmpeg"))
        self.ffprobe_path_var.set(settings.get("ffprobe_path", "ffprobe"))
        self.auto_dislike_var.set(settings.get("auto_dislike", "False") == "True")

        self.ffmpeg_manager.set_paths(self.ffmpeg_path_var.get(), self.ffprobe_path_var.get())

    def _save_settings(self) -> None:
        """Save current GUI settings."""

        self.settings_manager.save({
            "preview_seconds": self.preview_seconds_var.get(),
            "output_directory": self.output_directory_var.get(),
            "operation": self.operation_var.get(),
            "ffmpeg_path": self.ffmpeg_path_var.get(),
            "ffprobe_path": self.ffprobe_path_var.get(),
            "auto_dislike": str(bool(self.auto_dislike_var.get())),
        })

    # ------------------------------------------------------------------
    # Session
    # ------------------------------------------------------------------

    def _restore_session(self) -> None:
        """Restore the previous session, if one exists."""

        records = self.session_manager.load()

        if not records:
            self._update_counts()
            self._refresh_lists()
            return

        missing = 0

        for path, status in records:
            self.song_queue.add([path], status=status)

            song = Song(path=path)

            if status == STATUS_PENDING and not path.exists():
                self.song_queue.set_status(song, STATUS_SKIPPED)
                missing += 1

        LOGGER.info("Restored session with %d song(s), %d missing.", len(records), missing)

        summary = (
            f"Resumed previous session: {self.song_queue.remaining_count} remaining, "
            f"{len(self.song_queue.liked)} liked, {len(self.song_queue.disliked)} disliked."
        )

        if missing:
            summary += f" {missing} file(s) no longer exist and were marked skipped."

        self.status_var.set(summary)

        self._refresh_lists()
        self._update_counts()

        if self.song_queue.current_song is not None:
            self._show_current_song()

    def _schedule_session_save(self) -> None:
        """Save the session shortly after the latest change."""

        if self._session_save_job is not None:
            try:
                self.root.after_cancel(self._session_save_job)
            except Exception:
                pass

        self._session_save_job = self.root.after(SESSION_SAVE_DELAY_MS, self._save_session_now)

    def _save_session_now(self) -> None:
        """Write the session immediately."""

        self._session_save_job = None

        records = self.song_queue.records()

        if records:
            self.session_manager.save(records)
        else:
            self.session_manager.clear()

    # ------------------------------------------------------------------
    # GUI
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        """Build the complete user interface."""

        self._build_settings_panel()
        self._build_drop_panel()
        self._build_current_song_panel()
        self._build_action_panel()
        self._build_lists_panel()
        self._build_status_panel()

    def _build_settings_panel(self) -> None:
        """Build settings controls."""

        frame = ttk.LabelFrame(self.root, text="Settings", padding=10)
        frame.pack(fill="x", padx=10, pady=(10, 5))

        # Row 0.
        ttk.Label(frame, text="Preview length:").grid(row=0, column=0, sticky="w")

        ttk.Entry(frame, textvariable=self.preview_seconds_var, width=10).grid(
            row=0, column=1, sticky="w", padx=(5, 5))

        ttk.Label(frame, text="seconds").grid(row=0, column=2, sticky="w")

        ttk.Checkbutton(
            frame,
            text="No LIKE before the preview ends counts as DISLIKE",
            variable=self.auto_dislike_var,
            command=self._on_auto_dislike_changed,
        ).grid(row=0, column=3, sticky="w", padx=(20, 5))

        ttk.Label(frame, text="Operation:").grid(row=0, column=4, sticky="e", padx=(20, 5))

        ttk.Combobox(
            frame,
            textvariable=self.operation_var,
            values=("Copy", "Move"),
            state="readonly",
            width=8,
        ).grid(row=0, column=5, sticky="w")

        # Rows 1-3: paths.
        path_rows = (
            ("Output folder:", self.output_directory_var, self._choose_output_directory),
            ("FFmpeg:", self.ffmpeg_path_var, self._choose_ffmpeg),
            ("FFprobe:", self.ffprobe_path_var, self._choose_ffprobe),
        )

        for offset, (label, variable, command) in enumerate(path_rows, start=1):
            ttk.Label(frame, text=label).grid(row=offset, column=0, sticky="w", pady=(8, 0))

            ttk.Entry(frame, textvariable=variable).grid(
                row=offset, column=1, columnspan=5, sticky="ew", padx=(5, 5), pady=(8, 0))

            ttk.Button(frame, text="Browse...", command=command).grid(
                row=offset, column=6, padx=(5, 0), pady=(8, 0))

        for column in range(1, 6):
            frame.columnconfigure(column, weight=1)

    def _build_drop_panel(self) -> None:
        """Build the large drag-and-drop panel."""

        self.drop_frame = tk.Frame(self.root, relief="groove", borderwidth=3, height=200)
        self.drop_frame.pack(fill="x", padx=10, pady=8)
        self.drop_frame.pack_propagate(False)

        self.drop_label = tk.Label(
            self.drop_frame,
            text=(
                "DRAG AUDIO FILES OR FOLDERS HERE\n\n"
                "Folders are scanned recursively\n"
                "Non-audio files and duplicates are ignored\n\n"
                "(Click anywhere in this area to browse instead)"
            ),
            font=("TkDefaultFont", 16, "bold"),
            justify="center",
            cursor="hand2",
        )

        self.drop_label.pack(fill="both", expand=True)

        for widget in (self.drop_frame, self.drop_label):
            widget.bind("<Button-1>", lambda event: self._add_files_dialog())

    def _build_current_song_panel(self) -> None:
        """Build the current song display."""

        frame = ttk.LabelFrame(self.root, text="Currently Playing", padding=12)
        frame.pack(fill="x", padx=10, pady=5)

        ttk.Label(
            frame,
            textvariable=self.song_name_var,
            font=("TkDefaultFont", 18, "bold"),
            anchor="center",
        ).pack(fill="x")

        ttk.Label(frame, textvariable=self.song_path_var, anchor="center").pack(fill="x", pady=(3, 0))
        ttk.Label(frame, textvariable=self.position_var, anchor="center").pack(fill="x", pady=(5, 0))

        playback_frame = ttk.Frame(frame)
        playback_frame.pack(pady=(10, 0))

        self.play_button = ttk.Button(
            playback_frame, text="▶  Play / Restart Preview", command=self._play_current)
        self.play_button.pack(side="left", padx=(0, 5))

        self.pause_button = ttk.Button(
            playback_frame, text="⏸  Pause", command=self._pause_or_resume_current)
        self.pause_button.pack(side="left", padx=(5, 0))

    def _build_action_panel(self) -> None:
        """Build the large classification buttons."""

        frame = tk.Frame(self.root)
        frame.pack(fill="x", padx=20, pady=10)

        self.like_button = tk.Button(
            frame,
            text="👍  LIKE",
            font=("TkDefaultFont", 22, "bold"),
            height=2,
            command=lambda: self._classify(liked=True),
        )
        self.like_button.pack(side="left", fill="both", expand=True, padx=(0, 10))

        self.dislike_button = tk.Button(
            frame,
            text="👎  DISLIKE",
            font=("TkDefaultFont", 22, "bold"),
            height=2,
            command=lambda: self._classify(liked=False),
        )
        self.dislike_button.pack(side="left", fill="both", expand=True, padx=(10, 0))

        self._disable_action_buttons()

    def _build_lists_panel(self) -> None:
        """Build the All / Remaining / Liked / Disliked lists."""

        frame = ttk.LabelFrame(self.root, text="Songs", padding=10)
        frame.pack(fill="both", expand=True, padx=10, pady=5)

        toolbar = ttk.Frame(frame)
        toolbar.pack(fill="x", pady=(0, 5))

        ttk.Label(
            toolbar, textvariable=self.counts_var, font=("TkDefaultFont", 11, "bold")
        ).pack(side="left")

        buttons = (
            ("Clear Everything", self._clear_everything),
            ("Clear Queue", self._clear_queue),
            ("Clear Liked", self._clear_liked),
            ("Clear Disliked", self._clear_disliked),
            ("Export Liked", self._export_liked),
            ("Delete Disliked Files", self._delete_disliked_files),
        )

        for text, command in buttons:
            ttk.Button(toolbar, text=text, command=command).pack(side="right", padx=3)

        self.notebook = ttk.Notebook(frame)
        self.notebook.pack(fill="both", expand=True)

        self.list_boxes: dict[str, tk.Listbox] = {}
        self.list_tabs: dict[str, ttk.Frame] = {}

        for key, label in (
            ("all", "All"),
            ("remaining", "Remaining"),
            ("liked", "Liked"),
            ("disliked", "Disliked"),
        ):
            tab = ttk.Frame(self.notebook)
            self.notebook.add(tab, text=label)

            scrollbar = ttk.Scrollbar(tab, orient="vertical")
            scrollbar.pack(side="right", fill="y")

            listbox = tk.Listbox(tab, yscrollcommand=scrollbar.set, activestyle="dotbox")
            listbox.pack(side="left", fill="both", expand=True)

            scrollbar.configure(command=listbox.yview)

            listbox.bind("<Double-Button-1>", lambda event, name=key: self._requeue_selected(name))

            self.list_boxes[key] = listbox
            self.list_tabs[key] = tab

        ttk.Label(
            frame,
            text="Double-click a song in any list to send it back to the front of the queue.",
            foreground="#555555",
        ).pack(fill="x", pady=(5, 0))

    def _build_status_panel(self) -> None:
        """Build the status area."""

        frame = ttk.Frame(self.root, padding=(10, 5))
        frame.pack(fill="x")

        ttk.Label(frame, textvariable=self.status_var, anchor="w").pack(fill="x")

        ttk.Label(
            frame,
            text="Shortcuts:  ←  Like    →  Dislike    Space  Play / Pause",
            anchor="w",
            foreground="#555555",
        ).pack(fill="x")

    def _bind_shortcuts(self) -> None:
        """Bind keyboard shortcuts."""

        self.root.bind("<Left>", lambda event: self._shortcut(lambda: self._classify(liked=True)))
        self.root.bind("<Right>", lambda event: self._shortcut(lambda: self._classify(liked=False)))
        self.root.bind("<space>", lambda event: self._shortcut(self._toggle_playback))

    def _shortcut(self, action: Callable[[], None]) -> None:
        """Run a shortcut unless the user is typing in a text field."""

        widget = self.root.focus_get()

        if isinstance(widget, (tk.Entry, ttk.Entry, ttk.Combobox)):
            return

        action()

    def _toggle_playback(self) -> None:
        """Play the preview, or pause/resume if one is active."""

        if self.audio_player.is_playing() or self.audio_player.is_paused_state():
            self._pause_or_resume_current()
        else:
            self._play_current()

    def _on_auto_dislike_changed(self) -> None:
        """Persist the auto-dislike option when it changes."""

        self._save_settings()

        if self.auto_dislike_var.get():
            self.status_var.set("Auto-dislike is on. Press LIKE before the preview ends to keep a song.")
        else:
            self.status_var.set("Auto-dislike is off.")

    # ------------------------------------------------------------------
    # Drag and drop
    # ------------------------------------------------------------------

    def _configure_drag_and_drop(self) -> None:
        """Enable drag and drop if tkinterdnd2 is installed."""

        if tkinterdnd2 is None:
            self.drop_label.configure(
                text=(
                    "CLICK HERE TO ADD AUDIO FILES\n\n"
                    "Install tkinterdnd2 for drag-and-drop support"
                )
            )
            return

        try:
            for widget in (self.drop_label, self.drop_frame):
                widget.drop_target_register(DND_FILES)
                widget.dnd_bind("<<Drop>>", self._handle_drop)

        except Exception:
            LOGGER.exception("Could not configure drag-and-drop.")

    def _handle_drop(self, event: object) -> None:
        """Handle dropped files and folders."""

        try:
            paths = self.root.tk.splitlist(getattr(event, "data"))
            self._add_paths([Path(path) for path in paths])

        except Exception:
            LOGGER.exception("Could not process dropped files.")
            self.status_var.set("Could not process dropped files.")

    # ------------------------------------------------------------------
    # Adding files
    # ------------------------------------------------------------------

    def _add_files_dialog(self) -> None:
        """Open the audio file picker."""

        patterns = " ".join(f"*{extension}" for extension in sorted(AUDIO_EXTENSIONS))

        paths = filedialog.askopenfilenames(
            title="Add Audio Files",
            filetypes=[("Audio files", patterns), ("All files", "*.*")],
        )

        if not paths:
            return

        self._add_paths([Path(path) for path in paths])

    def _add_paths(self, paths: Sequence[Path]) -> None:
        """Discover and add audio files."""

        discovered = self.audio_file_manager.discover(paths)

        if not discovered:
            self.status_var.set("No supported audio files found.")
            return

        added = self.song_queue.add(discovered)
        ignored = len(discovered) - added

        self.status_var.set(
            f"Found {len(discovered)} audio file(s). Added {added}. Ignored {ignored} duplicate(s)."
        )

        LOGGER.info(
            "Audio discovery: found=%d, added=%d, duplicates=%d.",
            len(discovered), added, ignored,
        )

        self._refresh_lists()
        self._update_counts()
        self._schedule_session_save()

        # Analyze the first queued song so it is ready to play.
        if self.song_queue.current_song is not None and self.displayed_song is None:
            self._show_current_song()

    # ------------------------------------------------------------------
    # Current song
    # ------------------------------------------------------------------

    def _show_current_song(self) -> None:
        """Display and analyze the next song."""

        song = self.song_queue.current_song

        if song is None:
            self.audio_player.stop()

            self.current_analysis = None
            self.displayed_song = None

            self.song_name_var.set("No more songs to review")
            self.song_path_var.set("")
            self.position_var.set("")
            self.status_var.set("Queue finished. You can add more songs.")

            self._disable_action_buttons()
            self._refresh_lists()

            return

        self.current_song_token += 1
        self.current_analysis = None
        self.displayed_song = song

        self.song_name_var.set(song.name)
        self.song_path_var.set(str(song.path))
        self.position_var.set("")
        self.status_var.set(f"Analyzing: {song.name}")

        self._disable_action_buttons()
        self._refresh_lists()

        self._start_analysis(song, self.current_song_token)

    # ------------------------------------------------------------------
    # Analysis
    # ------------------------------------------------------------------

    def _get_preview_seconds(self) -> float:
        """Validate the preview duration."""

        try:
            value = float(self.preview_seconds_var.get())
        except ValueError as exc:
            raise ValueError("Preview length must be a number.") from exc

        if value <= 0:
            raise ValueError("Preview length must be greater than zero.")

        return value

    def _start_analysis(self, song: Song, token: int) -> None:
        """Start FFmpeg analysis on a background thread."""

        try:
            preview_seconds = self._get_preview_seconds()

        except ValueError as exc:
            messagebox.showerror(APPLICATION_NAME, str(exc))
            self._disable_action_buttons()
            return

        self.ffmpeg_manager.set_paths(self.ffmpeg_path_var.get(), self.ffprobe_path_var.get())

        self.is_analyzing = True
        self._disable_action_buttons()

        thread = threading.Thread(
            target=self._analysis_worker,
            args=(song, preview_seconds, token),
            daemon=True,
        )

        thread.start()

    def _analysis_worker(self, song: Song, preview_seconds: float, token: int) -> None:
        """Perform FFprobe and FFmpeg analysis."""

        try:
            duration = self.ffmpeg_manager.get_duration(song.path)

            def progress(analyzed_seconds: float) -> None:
                self.root.after(
                    0,
                    lambda: self._update_analysis_status(song, analyzed_seconds, duration, token),
                )

            active_start = self.ffmpeg_manager.find_most_active_section(
                path=song.path,
                preview_seconds=preview_seconds,
                duration_seconds=duration,
                progress_callback=progress,
            )

            result = AnalysisResult(duration_seconds=duration, active_start_seconds=active_start)

        except Exception as exc:
            LOGGER.exception("Audio analysis failed for %s.", song.path)
            self.root.after(0, lambda: self._analysis_failed(song, token, exc))
            return

        self.root.after(
            0, lambda: self._analysis_finished(song, token, result, preview_seconds))

    def _update_analysis_status(
        self, song: Song, analyzed_seconds: float, duration: float, token: int
    ) -> None:
        """Update analysis progress in the GUI."""

        if token != self.current_song_token:
            return

        percentage = min(100.0, analyzed_seconds / duration * 100.0) if duration > 0 else 0.0

        self.status_var.set(f"Analyzing {song.name}: {percentage:.0f}%")

    def _analysis_finished(
        self, song: Song, token: int, result: AnalysisResult, preview_seconds: float
    ) -> None:
        """Handle completed analysis."""

        if token != self.current_song_token:
            return

        self.is_analyzing = False
        self.current_analysis = result

        preview_end = min(
            result.duration_seconds, result.active_start_seconds + preview_seconds)

        self.position_var.set(
            f"Most active section: {result.active_start_seconds:.2f}s → {preview_end:.2f}s"
            f"  |  Song length: {result.duration_seconds:.2f}s"
        )

        self._enable_action_buttons()

        if not self.auto_play:
            self.status_var.set("Ready. Press Play to start continuous playback.")
            return

        try:
            self.audio_player.play(
                path=song.path,
                start_seconds=result.active_start_seconds,
                duration_seconds=preview_end - result.active_start_seconds,
            )

            self.pause_button.configure(text="⏸  Pause")
            self.status_var.set(self._playing_status())

        except Exception as exc:
            LOGGER.exception("Could not play %s.", song.path)
            self._skip_unplayable_song(song, token, exc)

    def _analysis_failed(self, song: Song, token: int, exc: Exception) -> None:
        """Handle an FFmpeg/FFprobe failure."""

        if token != self.current_song_token:
            return

        self.is_analyzing = False

        LOGGER.warning("Skipping song because analysis failed: %s | %s", song.path, exc)

        self.status_var.set(f"Could not analyze '{song.name}'. Skipping.")

        self._skip_current_without_classification(song)

        self.root.after(300, self._show_current_song)

    def _playing_status(self) -> str:
        """Return the status text shown while a preview plays."""

        if self.auto_dislike_var.get():
            return "Preview playing. Press LIKE to keep it, otherwise it will be disliked."

        return "Preview playing. Choose LIKE or DISLIKE."

    # ------------------------------------------------------------------
    # Playback
    # ------------------------------------------------------------------

    def _play_current(self) -> None:
        """Start continuous playback of the current song."""

        song = self.song_queue.current_song

        if song is None or self.is_analyzing:
            return

        if self.current_analysis is None:
            self.status_var.set("Please wait for audio analysis to finish.")
            return

        try:
            preview_seconds = self._get_preview_seconds()
        except ValueError as exc:
            messagebox.showerror(APPLICATION_NAME, str(exc))
            return

        preview_end = min(
            self.current_analysis.duration_seconds,
            self.current_analysis.active_start_seconds + preview_seconds,
        )

        try:
            # The user has started continuous playback.
            self.auto_play = True

            self.audio_player.play(
                path=song.path,
                start_seconds=self.current_analysis.active_start_seconds,
                duration_seconds=preview_end - self.current_analysis.active_start_seconds,
            )

            self.pause_button.configure(text="⏸  Pause")
            self.status_var.set(self._playing_status())

        except Exception as exc:
            LOGGER.exception("Could not play %s.", song.path)
            self._skip_unplayable_song(song, self.current_song_token, exc)

    def _pause_or_resume_current(self) -> None:
        """Pause or resume continuous playback."""

        if self.song_queue.current_song is None or self.is_analyzing:
            return

        if self.current_analysis is None:
            return

        if self.audio_player.is_paused_state():
            # Resuming also resumes automatic advancement.
            self.auto_play = True

            self.audio_player.resume()

            self.pause_button.configure(text="⏸  Pause")
            self.status_var.set(self._playing_status())

            return

        if self.audio_player.is_playing():
            # Pausing stops automatic advancement and auto-dislike.
            self.auto_play = False

            self.audio_player.pause()

            self.pause_button.configure(text="▶  Resume")
            self.status_var.set("Preview paused.")

    def _playback_timer(self) -> None:
        """Periodically check whether the preview has ended."""

        try:
            if self.audio_player.preview_expired():
                self.audio_player.stop()
                self.pause_button.configure(text="⏸  Pause")

                if self.song_queue.current_song is not None:
                    if self.auto_dislike_var.get() and self.auto_play:
                        self.status_var.set("No LIKE given. Disliking automatically.")
                        self._classify(liked=False, automatic=True)
                    else:
                        self.status_var.set("Preview finished. Choose LIKE or DISLIKE.")

        except Exception:
            LOGGER.exception("Playback timer encountered an error.")

        finally:
            self.root.after(PLAYBACK_CHECK_INTERVAL_MS, self._playback_timer)

    # ------------------------------------------------------------------
    # Classification
    # ------------------------------------------------------------------

    def _classify(self, liked: bool, automatic: bool = False) -> None:
        """Classify the currently displayed song."""

        if self.is_analyzing or self.song_queue.current_song is None:
            return

        self.audio_player.stop()

        classified = self.song_queue.classify_current(liked=liked)

        if classified is None:
            return

        self.current_song_token += 1
        self.current_analysis = None
        self.displayed_song = None

        verb = "Liked" if liked else ("Auto-disliked" if automatic else "Disliked")
        self.status_var.set(f"{verb}: {classified.name}")

        LOGGER.info("%s: %s", verb, classified.path)

        self._refresh_lists()
        self._update_counts()
        self._schedule_session_save()

        # Advance. If auto_play is on, _analysis_finished starts the next preview.
        self.root.after(100, self._show_current_song)

    def _skip_unplayable_song(self, song: Song, token: int, exc: Exception) -> None:
        """Skip a song that pygame cannot play."""

        if token != self.current_song_token:
            return

        self.audio_player.stop()

        LOGGER.warning("Skipping unplayable file %s: %s", song.path, exc)

        self.status_var.set(f"Could not play '{song.name}'. Skipping.")

        self._skip_current_without_classification(song)

        self.root.after(300, self._show_current_song)

    def _skip_current_without_classification(self, song: Song) -> None:
        """Mark a failed song as skipped instead of liked or disliked."""

        if self.song_queue.current_song != song:
            return

        self.song_queue.set_status(song, STATUS_SKIPPED)

        self.displayed_song = None
        self.current_analysis = None

        self._refresh_lists()
        self._update_counts()
        self._schedule_session_save()

    # ------------------------------------------------------------------
    # Buttons
    # ------------------------------------------------------------------

    def _enable_action_buttons(self) -> None:
        """Enable classification and playback controls."""

        for button in (self.like_button, self.dislike_button, self.play_button, self.pause_button):
            button.configure(state="normal")

    def _disable_action_buttons(self) -> None:
        """Disable classification and playback controls."""

        for button in (self.like_button, self.dislike_button, self.play_button, self.pause_button):
            button.configure(state="disabled")

        self.pause_button.configure(text="⏸  Pause")

    # ------------------------------------------------------------------
    # Counts and lists
    # ------------------------------------------------------------------

    def _update_counts(self) -> None:
        """Update the counts label."""

        self.counts_var.set(
            f"Total: {len(self.song_queue.songs)}    "
            f"Remaining: {self.song_queue.remaining_count}    "
            f"Liked: {len(self.song_queue.liked)}    "
            f"Disliked: {len(self.song_queue.disliked)}    "
            f"Skipped: {len(self.song_queue.skipped)}"
        )

    def _refresh_lists(self) -> None:
        """Rebuild every list from the single source of truth."""

        all_songs = list(self.song_queue.songs)
        remaining = self.song_queue.pending
        liked = self.song_queue.liked
        disliked = self.song_queue.disliked

        self._list_contents = {
            "all": all_songs,
            "remaining": remaining,
            "liked": liked,
            "disliked": disliked,
        }

        current = self.song_queue.current_song

        for key, songs in self._list_contents.items():
            listbox = self.list_boxes[key]
            selection = listbox.yview()[0]

            listbox.delete(0, tk.END)

            for song in songs:
                if key == "all":
                    listbox.insert(tk.END, f"[{self.song_queue.status_of(song).upper():<8}] {song.name}")
                else:
                    listbox.insert(tk.END, song.name)

            listbox.yview_moveto(selection)

        # Highlight the song being reviewed.
        if current is not None and remaining:
            self.list_boxes["remaining"].selection_clear(0, tk.END)
            self.list_boxes["remaining"].selection_set(0)

        labels = (
            ("all", "All", len(all_songs)),
            ("remaining", "Remaining", len(remaining)),
            ("liked", "Liked", len(liked)),
            ("disliked", "Disliked", len(disliked)),
        )

        for key, label, count in labels:
            self.notebook.tab(self.list_tabs[key], text=f"{label} ({count})")

    def _requeue_selected(self, list_key: str) -> None:
        """Send the double-clicked song back to the front of the queue."""

        listbox = self.list_boxes[list_key]
        selection = listbox.curselection()

        if not selection:
            return

        songs = getattr(self, "_list_contents", {}).get(list_key, [])
        index = selection[0]

        if index >= len(songs):
            return

        song = songs[index]

        if self.song_queue.status_of(song) == STATUS_PENDING and song == self.displayed_song:
            return

        self.song_queue.requeue(song)

        self.status_var.set(f"Requeued: {song.name}")

        self._refresh_lists()
        self._update_counts()
        self._schedule_session_save()

        if not self.is_analyzing:
            self._show_current_song()

    # ------------------------------------------------------------------
    # Clear operations
    # ------------------------------------------------------------------

    def _reset_current_display(self, message: str) -> None:
        """Stop playback and clear the current song display."""

        self.audio_player.stop()
        self.auto_play = False

        self.current_song_token += 1
        self.current_analysis = None
        self.displayed_song = None

        self.song_name_var.set("No song selected")
        self.song_path_var.set("")
        self.position_var.set("")
        self.status_var.set(message)

        self._disable_action_buttons()
        self._refresh_lists()
        self._update_counts()
        self._schedule_session_save()

    def _clear_queue(self) -> None:
        """Clear remaining songs."""

        if not self.song_queue.pending:
            return

        if not messagebox.askyesno(APPLICATION_NAME, "Clear the remaining songs?"):
            return

        self.song_queue.clear_pending()
        self._reset_current_display("Remaining songs cleared.")

    def _clear_liked(self) -> None:
        """Clear the liked list."""

        if not self.song_queue.liked:
            return

        if not messagebox.askyesno(APPLICATION_NAME, "Clear the liked list?"):
            return

        self.song_queue.clear_liked()

        self.status_var.set("Liked list cleared.")

        self._refresh_lists()
        self._update_counts()
        self._schedule_session_save()

    def _clear_disliked(self) -> None:
        """Clear the disliked list."""

        if not self.song_queue.disliked:
            return

        if not messagebox.askyesno(APPLICATION_NAME, "Clear the disliked list?"):
            return

        self.song_queue.clear_disliked()

        self.status_var.set("Disliked list cleared.")

        self._refresh_lists()
        self._update_counts()
        self._schedule_session_save()

    def _clear_everything(self) -> None:
        """Clear every list and the saved session."""

        if not messagebox.askyesno(
            APPLICATION_NAME,
            "Clear all songs and start a new session?\n\nThe saved session will be deleted.",
        ):
            return

        self.song_queue.clear_everything()
        self._reset_current_display("Everything was cleared.")
        self._save_session_now()

    # ------------------------------------------------------------------
    # Folder selection
    # ------------------------------------------------------------------

    def _choose_output_directory(self) -> None:
        """Select the output directory."""

        directory = filedialog.askdirectory(title="Select Output Folder")

        if not directory:
            return

        self.output_directory_var.set(directory)
        self._save_settings()

    def _choose_ffmpeg(self) -> None:
        """Select the FFmpeg executable."""

        path = filedialog.askopenfilename(
            title="Select FFmpeg Executable",
            filetypes=[("FFmpeg executable", "ffmpeg.exe"), ("All files", "*.*")],
        )

        if not path:
            return

        self.ffmpeg_path_var.set(path)
        self._save_settings()

    def _choose_ffprobe(self) -> None:
        """Select the FFprobe executable."""

        path = filedialog.askopenfilename(
            title="Select FFprobe Executable",
            filetypes=[("FFprobe executable", "ffprobe.exe"), ("All files", "*.*")],
        )

        if not path:
            return

        self.ffprobe_path_var.set(path)
        self._save_settings()

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def _export_liked(self) -> None:
        """Export all liked songs."""

        if self.is_exporting:
            return

        songs = self.song_queue.liked

        if not songs:
            messagebox.showinfo(APPLICATION_NAME, "There are no liked songs to export.")
            return

        output_text = self.output_directory_var.get().strip()

        if not output_text:
            self._choose_output_directory()
            output_text = self.output_directory_var.get().strip()

            if not output_text:
                return

        output_directory = Path(output_text).expanduser()

        move_files = self.operation_var.get() == "Move"
        operation = "Move" if move_files else "Copy"

        confirmed = messagebox.askyesno(
            APPLICATION_NAME,
            f"{operation} {len(songs)} liked song(s) to:\n\n{output_directory}\n\n"
            "SHA-256 will be used to detect identical existing files.\n\n"
            "Identical files will be skipped.\nDifferent files will be overwritten.",
        )

        if not confirmed:
            return

        if move_files:
            # Moving also needs the loaded file released.
            self.audio_player.release()
            self.auto_play = False

        self.is_exporting = True
        self.status_var.set("Exporting liked songs...")

        thread = threading.Thread(
            target=self._export_worker,
            args=(songs, output_directory, move_files),
            daemon=True,
        )

        thread.start()

    def _export_worker(
        self, songs: Sequence[Song], output_directory: Path, move_files: bool
    ) -> None:
        """Perform the export in the background."""

        def progress(index: int, total: int, filename: str) -> None:
            self.root.after(
                0, lambda: self.status_var.set(f"Exporting {index}/{total}: {filename}"))

        try:
            result = self.output_manager.export(
                songs=songs,
                output_directory=output_directory,
                move_files=move_files,
                progress_callback=progress,
            )

        except Exception as exc:
            LOGGER.exception("Unexpected export failure.")
            self.root.after(0, lambda: self._export_failed(exc))
            return

        self.root.after(0, lambda: self._export_finished(result))

    def _export_finished(self, result: ExportResult) -> None:
        """Display export results and update paths for moved files."""

        self.is_exporting = False

        for old_path, new_path in result.moved_pairs:
            self.song_queue.relocate(old_path, new_path)

        self.status_var.set(
            f"Export finished. Copied: {result.copied}, Moved: {result.moved}, "
            f"Skipped: {result.skipped}, Errors: {result.errors}"
        )

        message = (
            "Export complete.\n\n"
            f"Copied: {result.copied}\n"
            f"Moved: {result.moved}\n"
            f"Skipped: {result.skipped}\n"
            f"Errors: {result.errors}"
        )

        if result.error_messages:
            message += "\n\nFirst error:\n" + result.error_messages[0]

        self._refresh_lists()
        self._update_counts()
        self._schedule_session_save()

        messagebox.showinfo(APPLICATION_NAME, message)

        LOGGER.info(
            "Export finished: copied=%d, moved=%d, skipped=%d, errors=%d.",
            result.copied, result.moved, result.skipped, result.errors,
        )

    def _export_failed(self, exc: Exception) -> None:
        """Handle an unexpected export failure."""

        self.is_exporting = False

        self.status_var.set("Export failed.")

        messagebox.showerror(APPLICATION_NAME, f"The export failed unexpectedly:\n\n{exc}")

    # ------------------------------------------------------------------
    # Deleting disliked files
    # ------------------------------------------------------------------

    def _delete_disliked_files(self) -> None:
        """Delete every disliked song from disk."""

        if self.is_deleting or self.is_exporting:
            return

        songs = self.song_queue.disliked

        if not songs:
            messagebox.showinfo(APPLICATION_NAME, "There are no disliked songs.")
            return

        use_recycle_bin = send2trash is not None

        if use_recycle_bin:
            warning = "The files will be sent to the recycle bin."
        else:
            warning = (
                "The files will be deleted PERMANENTLY and cannot be recovered.\n\n"
                "Install send2trash if you would rather use the recycle bin."
            )

        preview = "\n".join(song.name for song in songs[:10])

        if len(songs) > 10:
            preview += f"\n...and {len(songs) - 10} more"

        if not messagebox.askyesno(
            APPLICATION_NAME,
            f"Delete {len(songs)} disliked file(s) from disk?\n\n{warning}\n\n{preview}",
            icon="warning",
        ):
            return

        if not use_recycle_bin and not messagebox.askyesno(
            APPLICATION_NAME,
            f"Last chance.\n\nPermanently delete {len(songs)} file(s)?",
            icon="warning",
            default="no",
        ):
            return

        # Release the loaded file. Pygame holds the last track open even when
        # stopped, which is what blocks deleting the song you just reviewed.
        self.audio_player.release()
        self.auto_play = False

        self.is_deleting = True
        self.status_var.set("Deleting disliked files...")

        thread = threading.Thread(
            target=self._delete_worker,
            args=(songs, use_recycle_bin),
            daemon=True,
        )

        thread.start()

    def _delete_worker(self, songs: Sequence[Song], use_recycle_bin: bool) -> None:
        """Perform the deletion in the background."""

        def progress(index: int, total: int, filename: str) -> None:
            self.root.after(
                0, lambda: self.status_var.set(f"Deleting {index}/{total}: {filename}"))

        try:
            result = self.output_manager.delete(
                songs=songs,
                use_recycle_bin=use_recycle_bin,
                progress_callback=progress,
            )

        except Exception as exc:
            LOGGER.exception("Unexpected delete failure.")
            self.root.after(0, lambda: self._delete_failed(exc))
            return

        self.root.after(0, lambda: self._delete_finished(result))

    def _delete_finished(self, result: DeleteResult) -> None:
        """Remove deleted songs from the lists and report the outcome."""

        self.is_deleting = False

        removed = {normalize_path(str(path)) for path in result.removed_paths}

        self.song_queue.remove(
            [song for song in self.song_queue.songs if song.key in removed])

        destination = "recycle bin" if result.used_recycle_bin else "disk"

        self.status_var.set(
            f"Delete finished. Removed from {destination}: {result.deleted}, "
            f"Already gone: {result.missing}, Errors: {result.errors}"
        )

        self._refresh_lists()
        self._update_counts()
        self._save_session_now()

        message = (
            "Delete complete.\n\n"
            f"Deleted: {result.deleted}\n"
            f"Already gone: {result.missing}\n"
            f"Errors: {result.errors}"
        )

        if result.error_messages:
            message += "\n\nFirst error:\n" + result.error_messages[0]
            message += "\n\nFiles that could not be deleted are still in the disliked list."

        messagebox.showinfo(APPLICATION_NAME, message)

        LOGGER.info(
            "Delete finished: deleted=%d, missing=%d, errors=%d.",
            result.deleted, result.missing, result.errors,
        )

    def _delete_failed(self, exc: Exception) -> None:
        """Handle an unexpected delete failure."""

        self.is_deleting = False

        self.status_var.set("Delete failed.")

        messagebox.showerror(APPLICATION_NAME, f"The delete failed unexpectedly:\n\n{exc}")

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------

    def _on_close(self) -> None:
        """Cleanly shut down the application."""

        LOGGER.info("Closing %s.", APPLICATION_NAME)

        self._save_settings()
        self._save_session_now()

        try:
            self.audio_player.shutdown()
        except Exception:
            LOGGER.exception("Error shutting down audio player.")

        self.root.destroy()


# ============================================================================
# Main
# ============================================================================

def main() -> None:
    """Start Song Curator."""

    LOGGER.info("Starting %s.", APPLICATION_NAME)

    root = tkinterdnd2.Tk() if tkinterdnd2 is not None else tk.Tk()

    SongCuratorApp(root)

    root.mainloop()


if __name__ == "__main__":
    main()