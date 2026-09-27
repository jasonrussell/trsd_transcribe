#!/usr/bin/env python3
"""Download transcripts from Timberlane Regional School District Vimeo videos.

Timberlane (TRSD.TV) posts its meetings on Vimeo at https://vimeo.com/trsd.
Unlike the Sandown CableCast setup, Vimeo's streams are not publicly fetchable:
the web client requires a logged-in account. This script therefore:

  1. Enumerates every video on the TRSD Vimeo channel via yt-dlp's `vimeo:user`
     extractor (which returns id + title for each video).
  2. Filters to meeting videos by matching known meeting-type keywords in the
     title (e.g. "TRSB Meeting", "Budget Committee").
  3. Downloads each selected video with yt-dlp, passing a cookie jar so the
     authenticated web client is used.
  4. Extracts audio and transcribes it with OpenAI whisper (same convention as
     sandown_transcribe), writing "<date>-<abbr>_transcript.log".

Authentication:
    yt-dlp needs cookies for Vimeo. Export them from your logged-in Vimeo
    browser session to a Netscape-format cookie file, then pass --cookies:

        python3 download_trsd.py --cookies ~/vimeo_cookies.txt

To export cookies from Chrome/Chromium (head must be running):
        # via yt-dlp's built-in browser-cookie support instead:
        python3 download_trsd.py --cookies-from-browser chrome

See https://github.com/yt-dlp/yt-dlp/wiki/FAQ#how-do-i-pass-cookies-to-yt-dlp
"""

import argparse
from datetime import date, datetime
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile


# Meeting types we care about. Each maps a set of title keywords to the short
# abbreviation used in transcript filenames (matching sandown_transcribe's
# conventions: 'bos', 'zba', etc.).
MEETING_PATTERNS = [
    (re.compile(r"\bbudget\s+committee\b", re.I), "budget"),
    (re.compile(r"\bboard\s+of\s+selectmen\b|\btrsb\b|trs\s*meeting\b", re.I), "bos"),
    (re.compile(r"planning\s+board\b|planning\s+committ\b", re.I), "planning"),
    (re.compile(r"\bzoning\s+board\b|zba\b", re.I), "zba"),
    (re.compile(r"\brecreation\b|\bremuneration\b", re.I), "recreation"),
]


def classify(title: str):
    """Return the meeting abbreviation for a video title, or None if not a meeting.

    Matches against known meeting-type keyword patterns. Returns None when the
    title looks like an event (concert, graduation, etc.) rather than a board
    or committee meeting.
    """
    for pattern, abbr in MEETING_PATTERNS:
        if pattern.search(title):
            return abbr
    return None


def extract_date_from_title(title: str):
    """Return a YYYYmmdd string parsed from a title's date, or None.

    Handles MM-DD-YYYY and YYYYMMDD forms found in TRSD titles (e.g.
    'TRSB Meeting 09/17/2026' or '20260625 TRSD ...').
    """
    # Explicit date at the end: "Meeting 09/17/2026"
    m = re.search(r"(\d{1,2})[-/.](\d{1,2})[-/.](\d{4})", title)
    if m:
        month, day, year = m.groups()
        try:
            return datetime(int(year), int(month), int(day)).strftime("%Y%m%d")
        except ValueError:
            pass
    # ISO-ish prefix: "20260625 TRSD ..."
    m = re.match(r"(\d{4})(\d{2})(\d{2})", title)
    if m:
        year, month, day = m.groups()
        try:
            return datetime(int(year), int(month), int(day)).strftime("%Y%m%d")
        except ValueError:
            pass
    return None


def list_videos(yt_dlp: str, cookies=None, cookies_from_browser=None):
    """Return a list of {id, title} dicts for every video on the TRSD channel.

    Uses yt-dlp's `vimeo:user` extractor with --flat to avoid resolving each
    video individually (fast). Returns entries sorted by id descending so the
    newest videos come first.
    """
    cmd = [yt_dlp, "--no-warnings", "--flat-playlist", "-J", "https://vimeo.com/trsd"]
    if cookies:
        cmd += ["--cookies", cookies]
    elif cookies_from_browser:
        cmd += ["--cookies-from-browser", cookies_from_browser]
    else:
        sys.exit("No authentication provided. Vimeo requires a logged-in account to "
                 "download. Use --cookies <file> or --cookies-from-browser chrome.")

    print("Enumerating TRSD channel via yt-dlp...")
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        sys.exit(f"yt-dlp enumeration failed:\n{proc.stderr}")

    data = json.loads(proc.stdout)
    entries = []
    for e in data.get("entries", []):
        title = e.get("title") or ""
        vid = e.get("id")
        if vid is None:
            continue
        entries.append({"id": str(vid), "title": title})

    entries.sort(key=lambda x: int(x["id"]), reverse=True)
    return entries


def download_video(yt_dlp: str, video_id: str, out_dir: str, cookies=None,
                   cookies_from_browser=None):
    """Download a single Vimeo video to out_dir using yt-dlp. Returns the file path."""
    os.makedirs(out_dir, exist_ok=True)
    # Vimeo serves separate audio-only and video-only streams (no combined
    # format), so requesting a video profile yields a file with no audio track.
    # We only need the audio for transcription; fall back to merging video+audio
    # if an audio-only stream is unavailable.
    cmd = [yt_dlp, "--no-warnings", "-f", "bestaudio/best",
           "-o", os.path.join(out_dir, f"{video_id}.%(ext)s"),
           f"https://vimeo.com/{video_id}"]
    if cookies:
        cmd += ["--cookies", cookies]
    elif cookies_from_browser:
        cmd += ["--cookies-from-browser", cookies_from_browser]

    print(f"  Downloading {video_id}...")
    proc = subprocess.run(cmd)
    if proc.returncode != 0:
        sys.exit(f"Download failed for video {video_id}.")

    # Find the downloaded file (yt-dlp may append a codec suffix to the extension).
    stem = f"{video_id}"
    matches = [f for f in os.listdir(out_dir)
               if f.startswith(stem + ".") or f == stem]
    if not matches:
        sys.exit(f"Could not locate downloaded file for video {video_id}.")
    return os.path.join(out_dir, matches[0])


def transcribe_video(video_path: str, log_dir: str, date_str: str | None, abbr: str,
                     model: str = "turbo", device: str | None = None) -> str:
    """Run whisper on a video file and write "<date>-<abbr>_transcript.log".

    ffmpeg extracts mono 16kHz audio; whisper writes an SRT which we combine into
    the project's .log format (whisper header + transcript body). Returns path.
    """
    if not shutil.which("ffmpeg"):
        sys.exit("ffmpeg not found; cannot extract audio for transcription.")

    base = f"{date_str}-{abbr}" if date_str else abbr
    out_path = os.path.join(log_dir, f"{base}.log")

    with tempfile.TemporaryDirectory(prefix="trsd_audio_") as tmp:
        audio_path = os.path.join(tmp, f"{base}.wav")
        cmd_ffmpeg = ["ffmpeg", "-y", "-loglevel", "error", "-i", video_path,
                      "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", audio_path]
        fferr = subprocess.run(cmd_ffmpeg, capture_output=True, text=True)
        if fferr.returncode != 0:
            sys.exit(f"ffmpeg could not extract an audio track from the video. "
                     f"(video file may be video-only or corrupt: {fferr.stderr.strip()})")

        print(f"  Transcribing with whisper -> {out_path}")
        captured = os.path.join(tmp, "whisper_capture.txt")
        cmd_whisper = ["whisper", audio_path, "--model", model,
                       "--output_dir", tmp, "--output_format", "srt"]
        if device:
            cmd_whisper += ["--device", device]
        else:
            # Default to CPU. The whisper CLI auto-selects CUDA when available, but
            # the desktop (Hyprland) already saturates VRAM (~15.6/16 GB), so loading
            # large-v3-turbo there OOMs. Use --device cuda only if explicitly requested.
            cmd_whisper += ["--device", "cpu"]
        with open(captured, "w") as cap:
            proc = subprocess.run(cmd_whisper, stdout=cap, stderr=subprocess.STDOUT)
        if proc.returncode != 0:
            tail = _captured_tail(captured)
            sys.exit(f"whisper failed with exit code {proc.returncode}. "
                     f"Last diagnostics:\n{tail}")

        srt_path = os.path.join(tmp, f"{base}.srt")
        if not os.path.exists(srt_path):
            sys.exit("whisper did not produce the expected transcript.")

        header = _extract_header(captured)
        with open(srt_path, encoding="utf-8") as fh:
            srt_text = fh.read().strip("\n")

        with open(out_path, "w", encoding="utf-8") as out:
            if header:
                out.write(header + "\n\n")
            out.write(srt_text + "\n")

    return out_path


def _captured_tail(captured_path: str, n: int = 15) -> str:
    """Return the last n non-empty lines of a captured file (for error reports)."""
    try:
        with open(captured_path, encoding="utf-8") as fh:
            lines = [ln for ln in fh.read().splitlines() if ln.strip()]
    except OSError:
        return "(could not read capture file)"
    tail = "\n".join(lines[-n:])
    return tail if tail else "(no output captured)"


def _extract_header(captured_path: str) -> str:
    """Return diagnostic lines (e.g. language detection) before the first SRT block."""
    with open(captured_path, encoding="utf-8") as fh:
        lines = fh.read().splitlines()
    end = 0
    for i, line in enumerate(lines):
        if re.match(r"\s*\[\d+:\d+", line):
            break
        end = i + 1
    return "\n".join(lines[:end]).strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cookies", default=None,
                        help="Path to a Netscape-format cookie jar for Vimeo login")
    parser.add_argument("--cookies-from-browser", default=None,
                        help="Browser name to pull cookies from (e.g. chrome)")
    parser.add_argument("--out", default="./downloads", help="Output directory")
    parser.add_argument("--log-dir", default=None,
                        help="Directory for transcript logs (default: same as --out)")
    parser.add_argument("--model", default="turbo", help="Whisper model name (default: turbo)")
    parser.add_argument("--device", default=None,
                        help="PyTorch device for whisper (cuda/cpu; default: cpu "
                             "to avoid VRAM contention with the desktop)")
    parser.add_argument("--from", dest="start", default=None, metavar="YYYY-MM-DD",
                        help="Only process videos on or after this date "
                             "(inclusive; use with --to for a range). "
                             "Date is parsed from each video's title.")
    parser.add_argument("--to", dest="end", default=None, metavar="YYYY-MM-DD",
                        help="Only process videos on or before this date "
                             "(inclusive; use with --from for a range).")
    parser.add_argument("--all", action="store_true",
                        help="Download every video regardless of title (not just meetings)")
    parser.add_argument("--keep-video", action="store_true",
                        help="Keep downloaded videos after transcription")
    parser.add_argument("--no-transcribe", action="store_true",
                        help="Skip whisper transcription and keep the video")
    args = parser.parse_args()

    # Resolve cookies (either explicit file or from a browser).
    cookies = args.cookies
    if not cookies and not args.cookies_from_browser:
        sys.exit("Authentication required. Export your logged-in Vimeo cookies and "
                 "pass --cookies ~/vimeo_cookies.txt\n"
                 "(See the script docstring for how to export them.)")

    yt_dlp = shutil.which("yt-dlp") or "yt-dlp"

    # Ensure whisper is importable; if not, give a helpful message.
    try:
        subprocess.run(["whisper", "--help"], capture_output=True, timeout=30)
    except FileNotFoundError:
        sys.exit("whisper not found on PATH. Source the openai-whisper venv first:\n"
                 "    source /home/nr4g3d/Code/openai-whisper/bin/activate")

    entries = list_videos(yt_dlp, cookies, args.cookies_from_browser)
    print(f"Found {len(entries)} videos on the TRSD channel.")

    # Filter to meetings unless --all.
    if args.all:
        selected = entries
    else:
        selected = []
        for e in entries:
            abbr = classify(e["title"])
            if abbr:
                e["abbr"] = abbr
                selected.append(e)
        print(f"{len(selected)} are board/committee meetings; "
              f"{len(entries) - len(selected)} skipped (events/concerts).")

    if not selected:
        sys.exit("No meeting videos found. Use --all to download everything.")

    # Apply optional date-range filter (dates parsed from each video's title).
    start = date.fromisoformat(args.start) if args.start else date.min
    end = date.fromisoformat(args.end) if args.end else date.max
    filtered = []
    for e in selected:
        dstr = extract_date_from_title(e["title"])
        if not dstr:
            # No parseable date. When a range is requested we can't confirm the
            # video falls inside it, so drop it; otherwise keep it as before.
            if args.start or args.end:
                print(f"  Skipping {e['title']}: no parseable date in range.")
                continue
            e["_date"] = None
            filtered.append(e)
            continue
        d = date.fromisoformat(dstr)
        if start <= d <= end:
            e["_date"] = dstr
            filtered.append(e)
        else:
            print(f"  Skipping {e['title']} ({d.isoformat()}): outside range "
                  f"{start.isoformat()}..{end.isoformat()}.")

    if not filtered:
        sys.exit("No meeting videos found in the requested date range.")
    skipped = len(selected) - len(filtered)
    if skipped:
        print(f"\n{skipped} meeting(s) outside the date range were skipped. "
              f"{len(filtered)} remain.")

    log_dir = args.log_dir or args.out
    os.makedirs(log_dir, exist_ok=True)

    for e in filtered:
        vid = e["id"]
        abbr = e.get("abbr") or classify(e["title"]) or "meeting"
        date_str = e.get("_date") or extract_date_from_title(e["title"])
        print(f"\n=== {e['title']} ({vid}) ===")

        try:
            video_path = download_video(yt_dlp, vid, args.out, cookies, args.cookies_from_browser)
        except SystemExit as ex:
            print(f"  Download failed: {ex}")
            continue

        if not args.no_transcribe:
            try:
                transcript = transcribe_video(video_path, log_dir, date_str, abbr, args.model, args.device)
                print(f"  Transcript saved to {transcript}")
            except SystemExit as ex:
                print(f"  Transcription failed: {ex}")

        if not args.keep_video:
            os.remove(video_path)
            print(f"  Removed video {video_path}.")


if __name__ == "__main__":
    main()
