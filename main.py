"""
Song Curator
============

Quickly listen to the most active portion of audio files and classify them
as liked or disliked.

Features
--------
- Drag and drop audio files and folders.
- Recursively scans dropped folders.
- Ignores non-audio files.
- Ignores duplicate files already in the session.
- Uses FFprobe to determine accurate audio duration.
- Uses FFmpeg to decode audio and find the most active X-second section.
- Configurable preview duration.
- Large LIKE / DISLIKE buttons.
- Automatically advances after classification.
- Additional files can be added while reviewing.
- Liked/disliked/remaining counts.
- Copy or move liked files to an output directory.
- SHA-256 comparison for existing output files.
- Existing files with the same SHA-256 are skipped.
- Existing files with a different SHA-256 are overwritten.
- FFmpeg / playback failures do not crash the application.
- Settings are saved to a JSON file and restored on startup.
- Logging to console and song_curator.log.

Requirements
------------
    pip install pygame tkinterdnd2

FFmpeg
-------
FFmpeg and FFprobe must either be available through PATH or specified in
the Settings section of the application.

Python
------
Python 3.10+
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import subprocess
import threading
import time
import wave

from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Sequence

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import pygame

try:
    import tkinterdnd2
    from tkinterdnd2 import DND_FILES
except ImportError:
    tkinterdnd2 = None
    DND_FILES = None


# ============================================================================
# Application constants
# ============================================================================

APPLICATION_NAME = "Song Curator"

SETTINGS_FILE_NAME = "song_curator_settings.json"
LOG_FILE_NAME = "song_curator.log"

DEFAULT_PREVIEW_SECONDS = 30.0

# Audio extensions accepted by the application.
AUDIO_EXTENSIONS = {
    ".mp3",
    ".m4a",
    ".flac",
    ".wav",
    ".ogg",
    ".oga",
    ".opus",
    ".aac",
    ".wma",
    ".aiff",
    ".aif",
    ".mp2",
    ".webm",
    ".mka",
}

# FFmpeg analysis settings.
#
# Audio is converted to mono 16-bit PCM at this sample rate.
ANALYSIS_SAMPLE_RATE = 16000

# Activity is calculated in these smaller blocks.
ACTIVITY_BLOCK_SECONDS = 0.25

# Number of seconds of silence/noise to allow around the requested
# preview window is handled naturally by the sliding window.
#
# SHA-256 read buffer.
HASH_BUFFER_SIZE = 1024 * 1024

# Export copy buffer.
COPY_BUFFER_SIZE = 1024 * 1024

# How often the GUI checks whether a preview should be stopped.
PLAYBACK_CHECK_INTERVAL_MS = 250


# ============================================================================
# Logging
# ============================================================================

def configure_logging() -> logging.Logger:
    """Configure application logging."""

    logger = logging.getLogger(APPLICATION_NAME)
    logger.setLevel(logging.INFO)

    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s"
    )

    file_handler = logging.FileHandler(
        LOG_FILE_NAME,
        encoding="utf-8",
    )

    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


LOGGER = configure_logging()


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
    error_messages: list[str] | None = None

    def __post_init__(self) -> None:
        if self.error_messages is None:
            self.error_messages = []


# ============================================================================
# Settings
# ============================================================================

class SettingsManager:
    """Loads and saves application settings as JSON."""

    def __init__(
        self,
        settings_path: Path,
    ) -> None:
        self.settings_path = settings_path

    def load(self) -> dict[str, str]:
        """Load settings from disk."""

        if not self.settings_path.exists():
            return {}

        try:
            with self.settings_path.open(
                "r",
                encoding="utf-8",
            ) as file:
                data = json.load(file)

            if not isinstance(data, dict):
                LOGGER.warning(
                    "Settings file did not contain a JSON object."
                )

                return {}

            result: dict[str, str] = {}

            for key, value in data.items():
                if isinstance(value, (str, int, float, bool)):
                    result[str(key)] = str(value)

            LOGGER.info(
                "Loaded settings from %s.",
                self.settings_path,
            )

            return result

        except Exception:
            LOGGER.exception(
                "Could not load settings file: %s",
                self.settings_path,
            )

            return {}

    def save(
        self,
        settings: dict[str, str],
    ) -> None:
        """Save settings to disk."""

        try:
            with self.settings_path.open(
                "w",
                encoding="utf-8",
            ) as file:
                json.dump(
                    settings,
                    file,
                    indent=4,
                )

            LOGGER.info(
                "Saved settings to %s.",
                self.settings_path,
            )

        except Exception:
            LOGGER.exception(
                "Could not save settings file: %s",
                self.settings_path,
            )


# ============================================================================
# Audio file discovery
# ============================================================================

class AudioFileManager:
    """Discovers supported audio files."""

    def __init__(
        self,
        extensions: set[str],
    ) -> None:
        self.extensions = {
            extension.lower()
            for extension in extensions
        }

    def is_audio_file(
        self,
        path: Path,
    ) -> bool:
        """Return whether a path is a supported audio file."""

        return (
            path.is_file()
            and path.suffix.lower()
            in self.extensions
        )

    def discover(
        self,
        paths: Sequence[Path],
    ) -> list[Path]:
        """
        Discover audio files.

        Files are accepted directly.

        Directories are recursively searched.
        """

        result: list[Path] = []

        for original_path in paths:
            try:
                path = (
                    original_path
                    .expanduser()
                    .resolve()
                )

            except OSError:
                LOGGER.exception(
                    "Could not resolve path: %s",
                    original_path,
                )

                continue

            if path.is_file():
                if self.is_audio_file(path):
                    result.append(path)

                continue

            if not path.is_dir():
                LOGGER.warning(
                    "Path does not exist or is not a directory: %s",
                    path,
                )

                continue

            try:
                for child in path.rglob("*"):
                    try:
                        if self.is_audio_file(child):
                            result.append(
                                child.resolve()
                            )

                    except OSError:
                        LOGGER.warning(
                            "Could not inspect: %s",
                            child,
                        )

            except OSError:
                LOGGER.exception(
                    "Could not scan directory: %s",
                    path,
                )

        return result


# ============================================================================
# FFmpeg
# ============================================================================

class FFmpegManager:
    """
    Handles FFmpeg and FFprobe operations.

    FFmpeg is used to decode audio into raw PCM for activity analysis.

    FFprobe is used to get accurate duration information.
    """

    def __init__(
        self,
        ffmpeg_path: str = "ffmpeg",
        ffprobe_path: str = "ffprobe",
    ) -> None:
        self.ffmpeg_path = ffmpeg_path
        self.ffprobe_path = ffprobe_path

    def set_paths(
        self,
        ffmpeg_path: str,
        ffprobe_path: str,
    ) -> None:
        """Update executable paths."""

        self.ffmpeg_path = (
            ffmpeg_path.strip()
            or "ffmpeg"
        )

        self.ffprobe_path = (
            ffprobe_path.strip()
            or "ffprobe"
        )

    def get_duration(
        self,
        path: Path,
    ) -> float:
        """Use FFprobe to obtain the audio duration."""

        command = [
            self.ffprobe_path,
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ]

        LOGGER.info(
            "Running FFprobe for duration: %s",
            path,
        )

        completed = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )

        if completed.returncode != 0:
            raise RuntimeError(
                (
                    "FFprobe failed with exit code "
                    f"{completed.returncode}: "
                    f"{completed.stderr.strip()}"
                )
            )

        output = completed.stdout.strip()

        try:
            duration = float(output)

        except ValueError as exc:
            raise RuntimeError(
                f"FFprobe returned an invalid duration: {output!r}"
            ) from exc

        if duration <= 0:
            raise RuntimeError(
                "FFprobe reported a duration of zero."
            )

        return duration

    def find_most_active_section(
        self,
        path: Path,
        preview_seconds: float,
        duration_seconds: float,
        progress_callback: Optional[
            Callable[[float], None]
        ] = None,
    ) -> float:
        """
        Find the most active contiguous section of the requested length.

        The audio is decoded by FFmpeg into mono 16-bit PCM.

        Audio is divided into small blocks and each block's RMS energy
        is calculated.

        A sliding window across those blocks determines which contiguous
        section has the highest total energy.

        Returns:
            Start time of the most active section.
        """

        if duration_seconds <= preview_seconds:
            return 0.0

        block_seconds = ACTIVITY_BLOCK_SECONDS

        command = [
            self.ffmpeg_path,
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(path),
            "-vn",
            "-sn",
            "-dn",
            "-ac",
            "1",
            "-ar",
            str(ANALYSIS_SAMPLE_RATE),
            "-f",
            "s16le",
            "pipe:1",
        ]

        LOGGER.info(
            "Analyzing activity with FFmpeg: %s",
            path,
        )

        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        if process.stdout is None:
            process.kill()

            raise RuntimeError(
                "Could not open FFmpeg stdout."
            )

        bytes_per_sample = 2

        bytes_per_block = int(
            ANALYSIS_SAMPLE_RATE
            * block_seconds
            * bytes_per_sample
        )

        if bytes_per_block <= 0:
            process.kill()

            raise RuntimeError(
                "Invalid FFmpeg analysis block size."
            )

        # Number of blocks needed to cover the requested preview.
        window_blocks = max(
            1,
            int(
                round(
                    preview_seconds
                    / block_seconds
                )
            ),
        )

        # Store (block_index, energy).
        energy_window: deque[
            tuple[int, float]
        ] = deque()

        energy_sum = 0.0

        best_energy = -1.0
        best_block_index = 0

        block_index = 0

        bytes_processed = 0

        try:
            while True:
                raw_block = process.stdout.read(
                    bytes_per_block
                )

                if not raw_block:
                    break

                complete_sample_bytes = (
                    len(raw_block)
                    // bytes_per_sample
                ) * bytes_per_sample

                raw_block = raw_block[
                    :complete_sample_bytes
                ]

                if not raw_block:
                    continue

                energy = self._calculate_rms(
                    raw_block
                )

                energy_window.append(
                    (
                        block_index,
                        energy,
                    )
                )

                energy_sum += energy

                if (
                    len(energy_window)
                    > window_blocks
                ):
                    _, old_energy = (
                        energy_window.popleft()
                    )

                    energy_sum -= old_energy

                if (
                    len(energy_window)
                    == window_blocks
                ):
                    if energy_sum > best_energy:
                        best_energy = energy_sum

                        best_block_index = (
                            energy_window[0][0]
                        )

                block_index += 1

                bytes_processed += len(raw_block)

                if progress_callback is not None:
                    estimated_seconds = (
                        bytes_processed
                        / (
                            ANALYSIS_SAMPLE_RATE
                            * bytes_per_sample
                        )
                    )

                    progress_callback(
                        estimated_seconds
                    )

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
            stderr_text = stderr_data.decode(
                "utf-8",
                errors="replace",
            ).strip()

            raise RuntimeError(
                (
                    f"FFmpeg failed with exit code "
                    f"{return_code}: {stderr_text}"
                )
            )

        if best_energy < 0:
            raise RuntimeError(
                "FFmpeg did not provide enough audio data for analysis."
            )

        active_start = (
            best_block_index
            * block_seconds
        )

        # Never allow the calculated start to leave the requested
        # preview extending beyond the end of the song.
        max_start = max(
            0.0,
            duration_seconds
            - preview_seconds,
        )

        return min(
            active_start,
            max_start,
        )

    @staticmethod
    def _calculate_rms(
        pcm_data: bytes,
    ) -> float:
        """Calculate normalized RMS energy from signed 16-bit PCM."""

        sample_count = (
            len(pcm_data)
            // 2
        )

        if sample_count <= 0:
            return 0.0

        total_square = 0.0

        # Iterate through signed little-endian 16-bit samples.
        for index in range(
            0,
            len(pcm_data) - 1,
            2,
        ):
            sample = int.from_bytes(
                pcm_data[
                    index:index + 2
                ],
                byteorder="little",
                signed=True,
            )

            normalized = (
                sample
                / 32768.0
            )

            total_square += (
                normalized
                * normalized
            )

        return (
            total_square
            / sample_count
        ) ** 0.5


# ============================================================================
# Audio player
# ============================================================================

class AudioPlayer:
    """Handles audio playback using pygame."""

    def __init__(self) -> None:
        self.initialized = False

        self.play_started_at = 0.0
        self.play_duration_seconds = 0.0

        # Amount of time that has elapsed before the most recent pause.
        self.elapsed_before_pause = 0.0

        self.is_paused = False

        self._initialize()

    def _initialize(self) -> None:
        """Initialize pygame mixer."""

        try:
            pygame.mixer.init()

            self.initialized = True

            LOGGER.info(
                "Pygame audio initialized."
            )

        except Exception:
            LOGGER.exception(
                "Could not initialize pygame audio."
            )

    def play(
        self,
        path: Path,
        start_seconds: float,
        duration_seconds: float,
    ) -> None:
        """Start or restart a preview from the specified position."""

        if not self.initialized:
            raise RuntimeError(
                "Pygame audio is not initialized."
            )

        self.stop()

        LOGGER.info(
            (
                "Playing %s from %.2f seconds "
                "for %.2f seconds.",
                path,
                start_seconds,
                duration_seconds,
            )
        )

        pygame.mixer.music.load(
            str(path)
        )

        try:
            pygame.mixer.music.play(
                loops=0,
                start=max(
                    0.0,
                    start_seconds,
                ),
            )

        except (TypeError, pygame.error):
            # Some pygame/SDL combinations have trouble with the
            # start parameter for certain codecs.
            #
            # Try normal playback and then seek.
            pygame.mixer.music.play()

            try:
                pygame.mixer.music.set_pos(
                    start_seconds
                )

            except Exception as exc:
                LOGGER.warning(
                    (
                        "Could not seek to %.2f seconds "
                        "for %s: %s",
                        start_seconds,
                        path,
                        exc,
                    )
                )

        self.play_started_at = time.monotonic()
        self.play_duration_seconds = max(
            0.0,
            duration_seconds,
        )

        self.elapsed_before_pause = 0.0
        self.is_paused = False

    def pause(self) -> None:
        """Pause the current preview."""

        if not self.initialized:
            return

        if not self.is_playing():
            return

        try:
            elapsed = (
                time.monotonic()
                - self.play_started_at
            )

            self.elapsed_before_pause += max(
                0.0,
                elapsed,
            )

            pygame.mixer.music.pause()

            self.is_paused = True

            LOGGER.info(
                "Playback paused."
            )

        except Exception:
            LOGGER.exception(
                "Could not pause playback."
            )

    def resume(self) -> None:
        """Resume paused playback."""

        if not self.initialized:
            return

        if not self.is_paused:
            return

        try:
            pygame.mixer.music.unpause()

            self.play_started_at = (
                time.monotonic()
            )

            self.is_paused = False

            LOGGER.info(
                "Playback resumed."
            )

        except Exception:
            LOGGER.exception(
                "Could not resume playback."
            )

    def stop(self) -> None:
        """Stop playback and reset playback state."""

        if not self.initialized:
            return

        try:
            pygame.mixer.music.stop()
        except Exception:
            LOGGER.exception(
                "Could not stop playback."
            )

        self.play_started_at = 0.0
        self.play_duration_seconds = 0.0
        self.elapsed_before_pause = 0.0
        self.is_paused = False

    def is_playing(self) -> bool:
        """Return whether pygame currently reports active playback."""

        if not self.initialized:
            return False

        if self.is_paused:
            return False

        try:
            return pygame.mixer.music.get_busy()
        except Exception:
            return False

    def is_paused_state(self) -> bool:
        """Return whether playback is currently paused."""

        return self.is_paused

    def preview_expired(self) -> bool:
        """Return whether the configured preview duration elapsed."""

        if self.play_started_at <= 0:
            return False

        if self.is_paused:
            return False

        elapsed = (
            time.monotonic()
            - self.play_started_at
            + self.elapsed_before_pause
        )

        return (
            elapsed
            >= self.play_duration_seconds
        )

    def shutdown(self) -> None:
        """Shut down the audio system."""

        try:
            self.stop()
            pygame.mixer.quit()
        except Exception:
            LOGGER.exception(
                "Could not shut down pygame."
            )


# ============================================================================
# SHA-256
# ============================================================================

class HashManager:
    """Calculates SHA-256 hashes."""

    @staticmethod
    def sha256(
        path: Path,
    ) -> str:
        """
        Calculate a SHA-256 hash.

        Files are read incrementally so very large audio files don't need
        to be loaded into memory.
        """

        digest = hashlib.sha256()

        with path.open("rb") as file:
            while True:
                chunk = file.read(
                    HASH_BUFFER_SIZE
                )

                if not chunk:
                    break

                digest.update(chunk)

        return digest.hexdigest()

    def are_identical(
        self,
        source: Path,
        destination: Path,
    ) -> bool:
        """
        Determine whether two files have identical contents.

        File size is checked first as a cheap optimization.
        SHA-256 is only calculated if the sizes match.
        """

        source_size = source.stat().st_size
        destination_size = destination.stat().st_size

        if source_size != destination_size:
            return False

        source_hash = self.sha256(
            source
        )

        destination_hash = self.sha256(
            destination
        )

        return source_hash == destination_hash


# ============================================================================
# Output manager
# ============================================================================

class OutputManager:
    """Copies or moves liked songs."""

    def __init__(
        self,
        hash_manager: HashManager,
    ) -> None:
        self.hash_manager = hash_manager

    def export(
        self,
        songs: Sequence[Song],
        output_directory: Path,
        move_files: bool,
        progress_callback: Optional[
            Callable[[int, int, str], None]
        ] = None,
    ) -> ExportResult:
        """Export songs to the output directory."""

        result = ExportResult()

        output_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        total = len(songs)

        for index, song in enumerate(
            songs,
            start=1,
        ):
            source = song.path

            destination = (
                output_directory
                / source.name
            )

            if progress_callback is not None:
                progress_callback(
                    index,
                    total,
                    source.name,
                )

            try:
                if not source.exists():
                    raise FileNotFoundError(
                        (
                            "Source file no longer exists: "
                            f"{source}"
                        )
                    )

                if destination.exists():
                    LOGGER.info(
                        (
                            "Destination exists. "
                            "Checking SHA-256: %s",
                            destination,
                        )
                    )

                    if self.hash_manager.are_identical(
                        source,
                        destination,
                    ):
                        result.skipped += 1

                        LOGGER.info(
                            (
                                "Skipping identical file: "
                                "%s",
                                destination,
                            )
                        )

                        continue

                    LOGGER.info(
                        (
                            "Destination differs from source. "
                            "It will be overwritten: %s",
                            destination,
                        )
                    )

                if move_files:
                    self._move(
                        source,
                        destination,
                    )

                    result.moved += 1

                else:
                    self._copy(
                        source,
                        destination,
                    )

                    result.copied += 1

            except Exception as exc:
                result.errors += 1

                message = (
                    f"{source.name}: {exc}"
                )

                result.error_messages.append(
                    message
                )

                LOGGER.exception(
                    "Could not export %s.",
                    source,
                )

        return result

    @staticmethod
    def _copy(
        source: Path,
        destination: Path,
    ) -> None:
        """Copy a file while preserving metadata."""

        with source.open("rb") as source_file:
            with destination.open("wb") as destination_file:

                while True:
                    chunk = source_file.read(
                        COPY_BUFFER_SIZE
                    )

                    if not chunk:
                        break

                    destination_file.write(
                        chunk
                    )

        shutil.copystat(
            source,
            destination,
        )

    @staticmethod
    def _move(
        source: Path,
        destination: Path,
    ) -> None:
        """Move a file."""

        if destination.exists():
            destination.unlink()

        shutil.move(
            str(source),
            str(destination),
        )


# ============================================================================
# Song queue
# ============================================================================

class SongQueue:
    """Maintains songs and their classification state."""

    def __init__(self) -> None:
        # Songs still waiting to be classified.
        self.pending: list[Song] = []

        # Songs already classified.
        self.liked: list[Song] = []
        self.disliked: list[Song] = []

        # Paths which have ever been added to this session.
        #
        # This means a song that was classified and then more files are
        # dragged in will still be recognized as a duplicate.
        self._known_paths: set[str] = set()

    @property
    def current_song(self) -> Optional[Song]:
        """Return the first pending song."""

        if not self.pending:
            return None

        return self.pending[0]

    @property
    def remaining_count(self) -> int:
        """Return the number of songs still waiting."""

        return len(self.pending)

    def add(
        self,
        paths: Sequence[Path],
    ) -> int:
        """Add songs while ignoring duplicates."""

        added = 0

        for path in paths:
            normalized = self._normalize(
                path
            )

            if normalized in self._known_paths:
                continue

            self._known_paths.add(
                normalized
            )

            self.pending.append(
                Song(path=path)
            )

            added += 1

        return added

    def classify_current(
        self,
        liked: bool,
    ) -> Optional[Song]:
        """Classify and remove the current song."""

        if not self.pending:
            return None

        song = self.pending.pop(0)

        if liked:
            self.liked.append(song)
        else:
            self.disliked.append(song)

        return song

    def clear_pending(self) -> None:
        """Clear pending songs."""

        for song in self.pending:
            self._known_paths.discard(
                self._normalize(song.path)
            )

        self.pending.clear()

    def clear_liked(self) -> None:
        """Clear liked songs."""

        self.liked.clear()

    def clear_disliked(self) -> None:
        """Clear disliked songs."""

        self.disliked.clear()

    def clear_everything(self) -> None:
        """Clear all lists."""

        self.pending.clear()
        self.liked.clear()
        self.disliked.clear()
        self._known_paths.clear()

    @staticmethod
    def _normalize(
        path: Path,
    ) -> str:
        """Normalize a path for duplicate detection."""

        try:
            return os.path.normcase(
                str(
                    path.resolve()
                )
            )

        except OSError:
            return os.path.normcase(
                str(path)
            )


# ============================================================================
# Main GUI
# ============================================================================

class SongCuratorApp:
    """Main Song Curator GUI."""

    def __init__(
        self,
        root: tk.Tk,
    ) -> None:
        self.root = root

        self.root.title(
            APPLICATION_NAME
        )

        self.root.geometry(
            "1150x800"
        )

        self.root.minsize(
            950,
            700,
        )

        # Managers.
        self.settings_manager = SettingsManager(
            Path(
                __file__
            ).resolve().parent
            / SETTINGS_FILE_NAME
        )

        self.audio_file_manager = (
            AudioFileManager(
                AUDIO_EXTENSIONS
            )
        )

        self.ffmpeg_manager = FFmpegManager()

        self.audio_player = AudioPlayer()

        self.hash_manager = HashManager()

        self.output_manager = OutputManager(
            self.hash_manager
        )

        self.song_queue = SongQueue()

        # Runtime state.
        self.current_analysis: Optional[
            AnalysisResult
        ] = None

        self.is_analyzing = False
        self.is_exporting = False

        # True when the user has started continuous playback.
        # Once enabled, classifying a song automatically starts the next one.
        self.auto_play = False

        self.current_song_token = 0

        # Variables.
        self.preview_seconds_var = (
            tk.StringVar()
        )

        self.output_directory_var = (
            tk.StringVar()
        )

        self.operation_var = (
            tk.StringVar()
        )

        self.ffmpeg_path_var = (
            tk.StringVar()
        )

        self.ffprobe_path_var = (
            tk.StringVar()
        )

        self.song_name_var = (
            tk.StringVar(
                value="No song selected"
            )
        )

        self.song_path_var = (
            tk.StringVar(
                value=""
            )
        )

        self.position_var = (
            tk.StringVar(
                value=""
            )
        )

        self.status_var = (
            tk.StringVar(
                value="Add audio files to begin."
            )
        )

        self.pending_count_var = (
            tk.StringVar(
                value="Remaining: 0"
            )
        )

        self.liked_count_var = (
            tk.StringVar(
                value="Liked: 0"
            )
        )

        self.disliked_count_var = (
            tk.StringVar(
                value="Disliked: 0"
            )
        )

        self._load_settings()

        self._build_ui()

        self._configure_drag_and_drop()

        self.root.protocol(
            "WM_DELETE_WINDOW",
            self._on_close,
        )

        self._update_counts()

        self.root.after(
            PLAYBACK_CHECK_INTERVAL_MS,
            self._playback_timer,
        )

    # ------------------------------------------------------------------
    # Settings
    # ------------------------------------------------------------------

    def _load_settings(self) -> None:
        """Load persisted settings."""

        settings = (
            self.settings_manager.load()
        )

        self.preview_seconds_var.set(
            settings.get(
                "preview_seconds",
                str(
                    DEFAULT_PREVIEW_SECONDS
                ),
            )
        )

        self.output_directory_var.set(
            settings.get(
                "output_directory",
                "",
            )
        )

        self.operation_var.set(
            settings.get(
                "operation",
                "Copy",
            )
        )

        self.ffmpeg_path_var.set(
            settings.get(
                "ffmpeg_path",
                "ffmpeg",
            )
        )

        self.ffprobe_path_var.set(
            settings.get(
                "ffprobe_path",
                "ffprobe",
            )
        )

        self.ffmpeg_manager.set_paths(
            self.ffmpeg_path_var.get(),
            self.ffprobe_path_var.get(),
        )

    def _save_settings(self) -> None:
        """Save current GUI settings."""

        settings = {
            "preview_seconds": (
                self.preview_seconds_var.get()
            ),
            "output_directory": (
                self.output_directory_var.get()
            ),
            "operation": (
                self.operation_var.get()
            ),
            "ffmpeg_path": (
                self.ffmpeg_path_var.get()
            ),
            "ffprobe_path": (
                self.ffprobe_path_var.get()
            ),
        }

        self.settings_manager.save(
            settings
        )

    # ------------------------------------------------------------------
    # GUI
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        """Build the complete user interface."""

        self._build_settings_panel()
        self._build_drop_panel()
        self._build_current_song_panel()
        self._build_action_panel()
        self._build_queue_panel()
        self._build_status_panel()

    def _build_settings_panel(self) -> None:
        """Build settings controls."""

        frame = ttk.LabelFrame(
            self.root,
            text="Settings",
            padding=10,
        )

        frame.pack(
            fill="x",
            padx=10,
            pady=(10, 5),
        )

        # Row 0.
        ttk.Label(
            frame,
            text="Preview length:",
        ).grid(
            row=0,
            column=0,
            sticky="w",
        )

        ttk.Entry(
            frame,
            textvariable=self.preview_seconds_var,
            width=10,
        ).grid(
            row=0,
            column=1,
            sticky="w",
            padx=(5, 20),
        )

        ttk.Label(
            frame,
            text="seconds",
        ).grid(
            row=0,
            column=2,
            sticky="w",
        )

        ttk.Label(
            frame,
            text="Operation:",
        ).grid(
            row=0,
            column=3,
            sticky="e",
            padx=(20, 5),
        )

        ttk.Combobox(
            frame,
            textvariable=self.operation_var,
            values=(
                "Copy",
                "Move",
            ),
            state="readonly",
            width=8,
        ).grid(
            row=0,
            column=4,
            sticky="w",
        )

        # Row 1.
        ttk.Label(
            frame,
            text="Output folder:",
        ).grid(
            row=1,
            column=0,
            sticky="w",
            pady=(8, 0),
        )

        ttk.Entry(
            frame,
            textvariable=self.output_directory_var,
        ).grid(
            row=1,
            column=1,
            columnspan=4,
            sticky="ew",
            padx=(5, 5),
            pady=(8, 0),
        )

        ttk.Button(
            frame,
            text="Browse...",
            command=self._choose_output_directory,
        ).grid(
            row=1,
            column=5,
            padx=(5, 0),
            pady=(8, 0),
        )

        # Row 2.
        ttk.Label(
            frame,
            text="FFmpeg:",
        ).grid(
            row=2,
            column=0,
            sticky="w",
            pady=(8, 0),
        )

        ttk.Entry(
            frame,
            textvariable=self.ffmpeg_path_var,
        ).grid(
            row=2,
            column=1,
            columnspan=4,
            sticky="ew",
            padx=(5, 5),
            pady=(8, 0),
        )

        ttk.Button(
            frame,
            text="Browse...",
            command=self._choose_ffmpeg,
        ).grid(
            row=2,
            column=5,
            padx=(5, 0),
            pady=(8, 0),
        )

        # Row 3.
        ttk.Label(
            frame,
            text="FFprobe:",
        ).grid(
            row=3,
            column=0,
            sticky="w",
            pady=(8, 0),
        )

        ttk.Entry(
            frame,
            textvariable=self.ffprobe_path_var,
        ).grid(
            row=3,
            column=1,
            columnspan=4,
            sticky="ew",
            padx=(5, 5),
            pady=(8, 0),
        )

        ttk.Button(
            frame,
            text="Browse...",
            command=self._choose_ffprobe,
        ).grid(
            row=3,
            column=5,
            padx=(5, 0),
            pady=(8, 0),
        )

        for column in range(1, 5):
            frame.columnconfigure(
                column,
                weight=1,
            )

    def _build_drop_panel(self) -> None:
        """Build the drag-and-drop panel."""

        self.drop_frame = tk.Frame(
            self.root,
            relief="groove",
            borderwidth=2,
            height=90,
        )

        self.drop_frame.pack(
            fill="x",
            padx=10,
            pady=5,
        )

        self.drop_label = tk.Label(
            self.drop_frame,
            text=(
                "DRAG AUDIO FILES OR FOLDERS HERE\n"
                "Non-audio files are ignored"
            ),
            font=(
                "TkDefaultFont",
                15,
                "bold",
            ),
            justify="center",
        )

        self.drop_label.pack(
            fill="both",
            expand=True,
            pady=20,
        )

        # Clicking the drop area also opens a file picker.
        self.drop_label.bind(
            "<Button-1>",
            lambda event: self._add_files_dialog(),
        )

    def _build_current_song_panel(self) -> None:
        """Build current song display."""

        frame = ttk.LabelFrame(
            self.root,
            text="Currently Playing",
            padding=15,
        )

        frame.pack(
            fill="x",
            padx=10,
            pady=5,
        )

        ttk.Label(
            frame,
            textvariable=self.song_name_var,
            font=(
                "TkDefaultFont",
                18,
                "bold",
            ),
            anchor="center",
        ).pack(
            fill="x",
        )

        ttk.Label(
            frame,
            textvariable=self.song_path_var,
            anchor="center",
        ).pack(
            fill="x",
            pady=(3, 0),
        )

        ttk.Label(
            frame,
            textvariable=self.position_var,
            anchor="center",
        ).pack(
            fill="x",
            pady=(5, 0),
        )

        playback_frame = ttk.Frame(
            frame,
        )

        playback_frame.pack(
            pady=(10, 0),
        )

        self.play_button = ttk.Button(
            playback_frame,
            text="▶  Play / Restart Preview",
            command=self._play_current,
        )

        self.play_button.pack(
            side="left",
            padx=(0, 5),
        )

        self.pause_button = ttk.Button(
            playback_frame,
            text="⏸  Pause",
            command=self._pause_or_resume_current,
        )

        self.pause_button.pack(
            side="left",
            padx=(5, 0),
        )

    def _build_action_panel(self) -> None:
        """Build large classification buttons."""

        frame = tk.Frame(
            self.root,
        )

        frame.pack(
            fill="x",
            padx=20,
            pady=10,
        )

        self.dislike_button = tk.Button(
            frame,
            text="👎  DISLIKE",
            font=(
                "TkDefaultFont",
                24,
                "bold",
            ),
            height=3,
            command=lambda: self._classify(
                liked=False
            ),
        )

        self.dislike_button.pack(
            side="left",
            fill="both",
            expand=True,
            padx=(0, 10),
        )

        self.like_button = tk.Button(
            frame,
            text="👍  LIKE",
            font=(
                "TkDefaultFont",
                24,
                "bold",
            ),
            height=3,
            command=lambda: self._classify(
                liked=True
            ),
        )

        self.like_button.pack(
            side="left",
            fill="both",
            expand=True,
            padx=(10, 0),
        )

    def _build_queue_panel(self) -> None:
        """Build queue and counts."""

        frame = ttk.LabelFrame(
            self.root,
            text="Queue",
            padding=10,
        )

        frame.pack(
            fill="both",
            expand=True,
            padx=10,
            pady=5,
        )

        counts_frame = ttk.Frame(
            frame,
        )

        counts_frame.pack(
            fill="x",
            pady=(0, 5),
        )

        ttk.Label(
            counts_frame,
            textvariable=self.pending_count_var,
            font=("TkDefaultFont", 11, "bold"),
        ).pack(
            side="left",
            padx=(0, 20),
        )

        ttk.Label(
            counts_frame,
            textvariable=self.liked_count_var,
            font=("TkDefaultFont", 11, "bold"),
        ).pack(
            side="left",
            padx=(0, 20),
        )

        ttk.Label(
            counts_frame,
            textvariable=self.disliked_count_var,
            font=("TkDefaultFont", 11, "bold"),
        ).pack(
            side="left",
        )

        ttk.Button(
            counts_frame,
            text="Export Liked",
            command=self._export_liked,
        ).pack(
            side="right",
            padx=3,
        )

        ttk.Button(
            counts_frame,
            text="Clear Disliked",
            command=self._clear_disliked,
        ).pack(
            side="right",
            padx=3,
        )

        ttk.Button(
            counts_frame,
            text="Clear Liked",
            command=self._clear_liked,
        ).pack(
            side="right",
            padx=3,
        )

        ttk.Button(
            counts_frame,
            text="Clear Queue",
            command=self._clear_queue,
        ).pack(
            side="right",
            padx=3,
        )

        ttk.Button(
            counts_frame,
            text="Clear Everything",
            command=self._clear_everything,
        ).pack(
            side="right",
            padx=3,
        )

        list_frame = ttk.Frame(
            frame,
        )

        list_frame.pack(
            fill="both",
            expand=True,
        )

        scrollbar = ttk.Scrollbar(
            list_frame,
            orient="vertical",
        )

        scrollbar.pack(
            side="right",
            fill="y",
        )

        self.queue_listbox = tk.Listbox(
            list_frame,
            yscrollcommand=scrollbar.set,
        )

        self.queue_listbox.pack(
            side="left",
            fill="both",
            expand=True,
        )

        scrollbar.configure(
            command=self.queue_listbox.yview
        )

    def _build_status_panel(self) -> None:
        """Build status area."""

        frame = ttk.Frame(
            self.root,
            padding=(10, 5),
        )

        frame.pack(
            fill="x",
        )

        ttk.Label(
            frame,
            textvariable=self.status_var,
            anchor="w",
        ).pack(
            fill="x",
        )

    # ------------------------------------------------------------------
    # Drag and drop
    # ------------------------------------------------------------------

    def _configure_drag_and_drop(self) -> None:
        """Enable drag and drop if tkinterdnd2 is installed."""

        if tkinterdnd2 is None:
            self.drop_label.configure(
                text=(
                    "CLICK HERE TO ADD AUDIO FILES\n"
                    "Install tkinterdnd2 for drag-and-drop"
                )
            )

            return

        try:
            self.drop_label.drop_target_register(
                DND_FILES
            )

            self.drop_label.dnd_bind(
                "<<Drop>>",
                self._handle_drop,
            )

            self.drop_frame.drop_target_register(
                DND_FILES
            )

            self.drop_frame.dnd_bind(
                "<<Drop>>",
                self._handle_drop,
            )

        except Exception:
            LOGGER.exception(
                "Could not configure drag-and-drop."
            )

    def _handle_drop(
        self,
        event: object,
    ) -> None:
        """Handle dropped files/folders."""

        try:
            raw_data = getattr(
                event,
                "data",
            )

            paths = self.root.tk.splitlist(
                raw_data
            )

            self._add_paths(
                [
                    Path(path)
                    for path in paths
                ]
            )

        except Exception:
            LOGGER.exception(
                "Could not process dropped files."
            )

            self.status_var.set(
                "Could not process dropped files."
            )

    # ------------------------------------------------------------------
    # Adding files
    # ------------------------------------------------------------------

    def _add_files_dialog(self) -> None:
        """Open audio file picker."""

        paths = filedialog.askopenfilenames(
            title="Add Audio Files",
            filetypes=[
                (
                    "Audio files",
                    " ".join(
                        f"*{extension}"
                        for extension
                        in sorted(
                            AUDIO_EXTENSIONS
                        )
                    ),
                ),
                (
                    "All files",
                    "*.*",
                ),
            ],
        )

        if not paths:
            return

        self._add_paths(
            [
                Path(path)
                for path in paths
            ]
        )

    def _add_paths(
        self,
        paths: Sequence[Path],
    ) -> None:
        """Discover and add audio files."""

        discovered = (
            self.audio_file_manager.discover(
                paths
            )
        )

        if not discovered:
            self.status_var.set(
                "No supported audio files found."
            )

            return

        added = self.song_queue.add(
            discovered
        )

        ignored = (
            len(discovered)
            - added
        )

        self.status_var.set(
            (
                f"Found {len(discovered)} audio file(s). "
                f"Added {added}. "
                f"Ignored {ignored} duplicate(s)."
            )
        )

        LOGGER.info(
            (
                "Audio discovery: found=%d, added=%d, "
                "duplicates=%d.",
                len(discovered),
                added,
                ignored,
            )
        )

        self._refresh_queue()
        self._update_counts()

        # Analyze the first queued song so it is ready to play.
        # Analysis does NOT start playback.
        if (
                self.song_queue.current_song is not None
                and not self.is_analyzing
                and self.current_analysis is None
        ):
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

            self.song_name_var.set(
                "No more songs to review"
            )

            self.song_path_var.set(
                ""
            )

            self.position_var.set(
                ""
            )

            self.status_var.set(
                (
                    "Queue finished. "
                    "You can add more songs."
                )
            )

            self._disable_action_buttons()

            return

        self.current_song_token += 1

        self.current_analysis = None

        self.song_name_var.set(
            song.name
        )

        self.song_path_var.set(
            str(song.path)
        )

        self.position_var.set(
            ""
        )

        self.status_var.set(
            f"Analyzing: {song.name}"
        )

        self._disable_action_buttons()

        self._start_analysis(
            song,
            self.current_song_token,
        )

    # ------------------------------------------------------------------
    # Analysis
    # ------------------------------------------------------------------

    def _get_preview_seconds(self) -> float:
        """Validate preview duration."""

        try:
            value = float(
                self.preview_seconds_var.get()
            )

        except ValueError as exc:
            raise ValueError(
                "Preview length must be a number."
            ) from exc

        if value <= 0:
            raise ValueError(
                "Preview length must be greater than zero."
            )

        return value

    def _start_analysis(
        self,
        song: Song,
        token: int,
    ) -> None:
        """Start FFmpeg analysis on a background thread."""

        if self.is_analyzing:
            return

        try:
            preview_seconds = (
                self._get_preview_seconds()
            )

        except ValueError as exc:
            messagebox.showerror(
                APPLICATION_NAME,
                str(exc),
            )

            self._disable_action_buttons()

            return

        self.ffmpeg_manager.set_paths(
            self.ffmpeg_path_var.get(),
            self.ffprobe_path_var.get(),
        )

        self.is_analyzing = True

        self._disable_action_buttons()

        thread = threading.Thread(
            target=self._analysis_worker,
            args=(
                song,
                preview_seconds,
                token,
            ),
            daemon=True,
        )

        thread.start()

    def _analysis_worker(
        self,
        song: Song,
        preview_seconds: float,
        token: int,
    ) -> None:
        """Perform FFprobe and FFmpeg analysis."""

        try:
            duration = (
                self.ffmpeg_manager.get_duration(
                    song.path
                )
            )

            def progress(
                analyzed_seconds: float,
            ) -> None:
                self.root.after(
                    0,
                    lambda: self._update_analysis_status(
                        song,
                        analyzed_seconds,
                        duration,
                        token,
                    ),
                )

            active_start = (
                self.ffmpeg_manager.find_most_active_section(
                    path=song.path,
                    preview_seconds=preview_seconds,
                    duration_seconds=duration,
                    progress_callback=progress,
                )
            )

            result = AnalysisResult(
                duration_seconds=duration,
                active_start_seconds=active_start,
            )

        except Exception as exc:
            LOGGER.exception(
                "Audio analysis failed for %s.",
                song.path,
            )

            self.root.after(
                0,
                lambda: self._analysis_failed(
                    song,
                    token,
                    exc,
                ),
            )

            return

        self.root.after(
            0,
            lambda: self._analysis_finished(
                song,
                token,
                result,
                preview_seconds,
            ),
        )

    def _update_analysis_status(
        self,
        song: Song,
        analyzed_seconds: float,
        duration: float,
        token: int,
    ) -> None:
        """Update analysis progress in the GUI."""

        if token != self.current_song_token:
            return

        if self.song_queue.current_song != song:
            return

        percentage = 0.0

        if duration > 0:
            percentage = min(
                100.0,
                (
                    analyzed_seconds
                    / duration
                )
                * 100.0,
            )

        self.status_var.set(
            (
                f"Analyzing {song.name}: "
                f"{percentage:.0f}%"
            )
        )

    def _analysis_finished(
        self,
        song: Song,
        token: int,
        result: AnalysisResult,
        preview_seconds: float,
    ) -> None:
        """Handle completed analysis."""

        self.is_analyzing = False

        if token != self.current_song_token:
            return

        if self.song_queue.current_song != song:
            return

        self.current_analysis = result

        preview_end = min(
            result.duration_seconds,
            result.active_start_seconds
            + preview_seconds,
        )

        self.position_var.set(
            (
                f"Most active section: "
                f"{result.active_start_seconds:.2f}s "
                f"→ {preview_end:.2f}s  |  "
                f"Song length: "
                f"{result.duration_seconds:.2f}s"
            )
        )

        if self.auto_play:
            try:
                self.audio_player.play(
                    path=song.path,
                    start_seconds=(
                        result.active_start_seconds
                    ),
                    duration_seconds=(
                            preview_end
                            - result.active_start_seconds
                    ),
                )

                self.pause_button.configure(
                    text="⏸  Pause"
                )

                self.status_var.set(
                    "Preview playing. Choose LIKE or DISLIKE."
                )

            except Exception as exc:
                LOGGER.exception(
                    "Could not play %s.",
                    song.path,
                )

                self._skip_unplayable_song(
                    song,
                    token,
                    exc,
                )

                return

        else:
            self.status_var.set(
                "Ready. Press Play to start continuous playback."
            )

        self._enable_action_buttons()

    def _analysis_failed(
        self,
        song: Song,
        token: int,
        exc: Exception,
    ) -> None:
        """Handle an FFmpeg/FFprobe failure."""

        self.is_analyzing = False

        if token != self.current_song_token:
            return

        LOGGER.warning(
            (
                "Skipping song because analysis failed: "
                "%s | %s",
                song.path,
                exc,
            )
        )

        self.status_var.set(
            (
                f"Could not analyze '{song.name}'. "
                "Skipping."
            )
        )

        self._skip_current_without_classification(
            song
        )

        self.root.after(
            300,
            self._show_current_song,
        )

    # ------------------------------------------------------------------
    # Playback
    # ------------------------------------------------------------------

    def _pause_or_resume_current(self) -> None:
        """Pause or resume continuous playback."""

        song = self.song_queue.current_song

        if song is None:
            return

        if self.is_analyzing:
            return

        if self.current_analysis is None:
            return

        if self.audio_player.is_paused_state():
            # Resume playback and continue automatic advancement.
            self.auto_play = True

            self.audio_player.resume()

            self.pause_button.configure(
                text="⏸  Pause"
            )

            self.status_var.set(
                "Preview resumed. Choose LIKE or DISLIKE."
            )

            return

        if self.audio_player.is_playing():
            # Pausing also disables automatic advancement.
            self.auto_play = False

            self.audio_player.pause()

            self.pause_button.configure(
                text="▶  Resume"
            )

            self.status_var.set(
                "Preview paused."
            )

    def _play_current(self) -> None:
        """Start continuous playback of the current song."""

        song = self.song_queue.current_song

        if song is None:
            return

        if self.is_analyzing:
            return

        if self.current_analysis is None:
            self.status_var.set(
                "Please wait for audio analysis to finish."
            )

            return

        try:
            preview_seconds = (
                self._get_preview_seconds()
            )

        except ValueError as exc:
            messagebox.showerror(
                APPLICATION_NAME,
                str(exc),
            )

            return

        preview_end = min(
            self.current_analysis.duration_seconds,
            self.current_analysis.active_start_seconds
            + preview_seconds,
        )

        try:
            # Tell the application that the user has started
            # continuous playback.
            self.auto_play = True

            self.audio_player.play(
                path=song.path,
                start_seconds=(
                    self.current_analysis.active_start_seconds
                ),
                duration_seconds=(
                        preview_end
                        - self.current_analysis.active_start_seconds
                ),
            )

            self.pause_button.configure(
                text="⏸  Pause"
            )

            self.status_var.set(
                "Preview playing. Choose LIKE or DISLIKE."
            )

        except Exception as exc:
            LOGGER.exception(
                "Could not play %s.",
                song.path,
            )

            self._skip_unplayable_song(
                song,
                self.current_song_token,
                exc,
            )


    def _pause_or_resume_current(self) -> None:
        """Pause or resume the current preview."""

        song = self.song_queue.current_song

        if song is None:
            return

        if self.is_analyzing:
            return

        if self.current_analysis is None:
            return

        if self.audio_player.is_paused_state():
            self.audio_player.resume()

            self.pause_button.configure(
                text="⏸  Pause"
            )

            self.status_var.set(
                "Preview resumed. Choose LIKE or DISLIKE."
            )

            return

        if self.audio_player.is_playing():
            self.audio_player.pause()

            self.pause_button.configure(
                text="▶  Resume"
            )

            self.status_var.set(
                "Preview paused."
            )

    def _playback_timer(self) -> None:
        """Periodically check whether the preview has ended."""

        try:
            if self.audio_player.preview_expired():
                self.audio_player.stop()

                self.pause_button.configure(
                    text="⏸  Pause"
                )

                if self.song_queue.current_song is not None:
                    self.status_var.set(
                        (
                            "Preview finished. "
                            "Choose LIKE or DISLIKE."
                        )
                    )

        except Exception:
            LOGGER.exception(
                "Playback timer encountered an error."
            )

        finally:
            self.root.after(
                PLAYBACK_CHECK_INTERVAL_MS,
                self._playback_timer,
            )

    # ------------------------------------------------------------------
    # Classification
    # ------------------------------------------------------------------

    def _classify(
            self,
            liked: bool,
    ) -> None:
        """Classify the currently displayed song."""

        if self.is_analyzing:
            return

        song = self.song_queue.current_song

        if song is None:
            return

        # Stop the current preview immediately.
        self.audio_player.stop()

        classified = (
            self.song_queue.classify_current(
                liked=liked
            )
        )

        if classified is None:
            return

        self.current_song_token += 1

        self.current_analysis = None

        if liked:
            self.status_var.set(
                f"Liked: {classified.name}"
            )
        else:
            self.status_var.set(
                f"Disliked: {classified.name}"
            )

        self._refresh_queue()
        self._update_counts()

        # Automatically advance to the next song.
        #
        # If auto_play is True, _analysis_finished() will
        # automatically start the next preview.
        self.root.after(
            100,
            self._show_current_song,
        )

    def _skip_unplayable_song(
        self,
        song: Song,
        token: int,
        exc: Exception,
    ) -> None:
        """Skip a song that Python/pygame cannot play."""

        if token != self.current_song_token:
            return

        self.audio_player.stop()

        LOGGER.warning(
            (
                "Skipping unplayable file %s: %s",
                song.path,
                exc,
            )
        )

        self.status_var.set(
            (
                f"Could not play '{song.name}'. "
                "Skipping."
            )
        )

        self._skip_current_without_classification(
            song
        )

        self.root.after(
            300,
            self._show_current_song,
        )

    def _skip_current_without_classification(
        self,
        song: Song,
    ) -> None:
        """
        Remove a failed song from the pending queue without putting it
        into liked or disliked.
        """

        current = (
            self.song_queue.current_song
        )

        if current != song:
            return

        if self.song_queue.pending:
            self.song_queue.pending.pop(0)

        self._update_counts()
        self._refresh_queue()

    # ------------------------------------------------------------------
    # Buttons
    # ------------------------------------------------------------------

    def _enable_action_buttons(self) -> None:
        """Enable classification and playback controls."""

        self.like_button.configure(
            state="normal"
        )

        self.dislike_button.configure(
            state="normal"
        )

        self.play_button.configure(
            state="normal"
        )

        self.pause_button.configure(
            state="normal"
        )

    def _disable_action_buttons(self) -> None:
        """Disable classification and playback controls."""

        self.like_button.configure(
            state="disabled"
        )

        self.dislike_button.configure(
            state="disabled"
        )

        self.play_button.configure(
            state="disabled"
        )

        self.pause_button.configure(
            state="disabled"
        )

        self.pause_button.configure(
            text="⏸  Pause"
        )

    # ------------------------------------------------------------------
    # Counts and queue
    # ------------------------------------------------------------------

    def _update_counts(self) -> None:
        """Update queue counts."""

        self.pending_count_var.set(
            (
                f"Remaining: "
                f"{self.song_queue.remaining_count}"
            )
        )

        self.liked_count_var.set(
            (
                f"Liked: "
                f"{len(self.song_queue.liked)}"
            )
        )

        self.disliked_count_var.set(
            (
                f"Disliked: "
                f"{len(self.song_queue.disliked)}"
            )
        )

    def _refresh_queue(self) -> None:
        """Refresh queue list."""

        self.queue_listbox.delete(
            0,
            tk.END,
        )

        for song in self.song_queue.pending:
            self.queue_listbox.insert(
                tk.END,
                song.name,
            )

    # ------------------------------------------------------------------
    # Clear operations
    # ------------------------------------------------------------------

    def _clear_queue(self) -> None:
        """Clear pending queue."""

        self.audio_player.stop()
        self.auto_play = False

        self.current_song_token += 1
        self.current_analysis = None

        self.song_queue.clear_pending()

        self.song_name_var.set(
            "No song selected"
        )

        self.song_path_var.set(
            ""
        )

        self.position_var.set(
            ""
        )

        self.status_var.set(
            "Queue cleared."
        )

        self._disable_action_buttons()
        self._refresh_queue()
        self._update_counts()

    def _clear_liked(self) -> None:
        """Clear liked list."""

        if not self.song_queue.liked:
            return

        if not messagebox.askyesno(
            APPLICATION_NAME,
            "Clear the liked list?",
        ):
            return

        self.song_queue.clear_liked()

        self.status_var.set(
            "Liked list cleared."
        )

        self._update_counts()

    def _clear_disliked(self) -> None:
        """Clear disliked list."""

        if not self.song_queue.disliked:
            return

        if not messagebox.askyesno(
            APPLICATION_NAME,
            "Clear the disliked list?",
        ):
            return

        self.song_queue.clear_disliked()

        self.status_var.set(
            "Disliked list cleared."
        )

        self._update_counts()

    def _clear_everything(self) -> None:
        """Clear every list."""

        if not messagebox.askyesno(
            APPLICATION_NAME,
            (
                "Clear the queue, liked list, "
                "and disliked list?"
            ),
        ):
            return

        self.audio_player.stop()
        self.auto_play = False

        self.current_song_token += 1
        self.current_analysis = None

        self.song_queue.clear_everything()

        self.song_name_var.set(
            "No song selected"
        )

        self.song_path_var.set(
            ""
        )

        self.position_var.set(
            ""
        )

        self.status_var.set(
            "Everything was cleared."
        )

        self._disable_action_buttons()
        self._refresh_queue()
        self._update_counts()

    # ------------------------------------------------------------------
    # Folder selection
    # ------------------------------------------------------------------

    def _choose_output_directory(self) -> None:
        """Select output directory."""

        directory = filedialog.askdirectory(
            title="Select Output Folder"
        )

        if not directory:
            return

        self.output_directory_var.set(
            directory
        )

        self._save_settings()

    def _choose_ffmpeg(self) -> None:
        """Select ffmpeg executable."""

        path = filedialog.askopenfilename(
            title="Select FFmpeg Executable",
            filetypes=[
                (
                    "FFmpeg executable",
                    "ffmpeg.exe",
                ),
                (
                    "All files",
                    "*.*",
                ),
            ],
        )

        if not path:
            return

        self.ffmpeg_path_var.set(
            path
        )

        self._save_settings()

    def _choose_ffprobe(self) -> None:
        """Select ffprobe executable."""

        path = filedialog.askopenfilename(
            title="Select FFprobe Executable",
            filetypes=[
                (
                    "FFprobe executable",
                    "ffprobe.exe",
                ),
                (
                    "All files",
                    "*.*",
                ),
            ],
        )

        if not path:
            return

        self.ffprobe_path_var.set(
            path
        )

        self._save_settings()

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def _export_liked(self) -> None:
        """Export all liked songs."""

        if self.is_exporting:
            return

        songs = list(
            self.song_queue.liked
        )

        if not songs:
            messagebox.showinfo(
                APPLICATION_NAME,
                "There are no liked songs to export.",
            )

            return

        output_text = (
            self.output_directory_var
            .get()
            .strip()
        )

        if not output_text:
            self._choose_output_directory()

            output_text = (
                self.output_directory_var
                .get()
                .strip()
            )

            if not output_text:
                return

        output_directory = Path(
            output_text
        ).expanduser()

        move_files = (
            self.operation_var.get()
            == "Move"
        )

        operation = (
            "move"
            if move_files
            else "copy"
        )

        confirmed = messagebox.askyesno(
            APPLICATION_NAME,
            (
                f"{operation.title()} "
                f"{len(songs)} liked song(s) to:\n\n"
                f"{output_directory}\n\n"
                f"SHA-256 will be used to detect "
                f"identical existing files.\n\n"
                f"Identical files will be skipped.\n"
                f"Different files will be overwritten."
            ),
        )

        if not confirmed:
            return

        self.is_exporting = True

        self.status_var.set(
            "Exporting liked songs..."
        )

        thread = threading.Thread(
            target=self._export_worker,
            args=(
                songs,
                output_directory,
                move_files,
            ),
            daemon=True,
        )

        thread.start()

    def _export_worker(
        self,
        songs: Sequence[Song],
        output_directory: Path,
        move_files: bool,
    ) -> None:
        """Perform export in background."""

        def progress(
            index: int,
            total: int,
            filename: str,
        ) -> None:
            self.root.after(
                0,
                lambda: self.status_var.set(
                    (
                        f"Exporting "
                        f"{index}/{total}: "
                        f"{filename}"
                    )
                ),
            )

        try:
            result = (
                self.output_manager.export(
                    songs=songs,
                    output_directory=output_directory,
                    move_files=move_files,
                    progress_callback=progress,
                )
            )

        except Exception as exc:
            LOGGER.exception(
                "Unexpected export failure."
            )

            self.root.after(
                0,
                lambda: self._export_failed(
                    exc
                ),
            )

            return

        self.root.after(
            0,
            lambda: self._export_finished(
                result
            ),
        )

    def _export_finished(
        self,
        result: ExportResult,
    ) -> None:
        """Display export results."""

        self.is_exporting = False

        self.status_var.set(
            (
                f"Export finished. "
                f"Copied: {result.copied}, "
                f"Moved: {result.moved}, "
                f"Skipped: {result.skipped}, "
                f"Errors: {result.errors}"
            )
        )

        message = (
            "Export complete.\n\n"
            f"Copied: {result.copied}\n"
            f"Moved: {result.moved}\n"
            f"Skipped: {result.skipped}\n"
            f"Errors: {result.errors}"
        )

        if result.error_messages:
            message += (
                "\n\nFirst error:\n"
                + result.error_messages[0]
            )

        messagebox.showinfo(
            APPLICATION_NAME,
            message,
        )

        LOGGER.info(
            (
                "Export finished: copied=%d, moved=%d, "
                "skipped=%d, errors=%d.",
                result.copied,
                result.moved,
                result.skipped,
                result.errors,
            )
        )

    def _export_failed(
        self,
        exc: Exception,
    ) -> None:
        """Handle an unexpected export failure."""

        self.is_exporting = False

        self.status_var.set(
            "Export failed."
        )

        messagebox.showerror(
            APPLICATION_NAME,
            (
                "The export failed unexpectedly:\n\n"
                f"{exc}"
            ),
        )

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------

    def _on_close(self) -> None:
        """Cleanly shut down the application."""

        LOGGER.info(
            "Closing %s.",
            APPLICATION_NAME,
        )

        self._save_settings()

        try:
            self.audio_player.shutdown()
        except Exception:
            LOGGER.exception(
                "Error shutting down audio player."
            )

        self.root.destroy()


# ============================================================================
# Main
# ============================================================================

def main() -> None:
    """Start Song Curator."""

    LOGGER.info(
        "Starting %s.",
        APPLICATION_NAME,
    )

    if tkinterdnd2 is not None:
        root = tkinterdnd2.Tk()
    else:
        root = tk.Tk()

    SongCuratorApp(root)

    root.mainloop()


if __name__ == "__main__":
    main()