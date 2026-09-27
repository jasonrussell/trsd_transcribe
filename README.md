# TRSD Transcribe

Transcript extraction for **Timberlane Regional School District** (TRSD.TV) 

## How it works

Timberlane posts its meetings on [Vimeo](https://vimeo.com/trsd). Vimeo's streams are **not publicly
fetchable** — the web client requires a logged-in account. This script therefore:

1. Enumerates every video on the TRSD channel via yt-dlp's `vimeo:user` extractor.
2. Filters to board/committee meetings by matching title keywords (`TRSB Meeting`,
   `Budget Committee`, `Planning Board`, `Zoning Board`, etc.). Events and concerts
   are skipped unless you pass `--all`.
3. Downloads each selected video with yt-dlp, passing a cookie jar so the
   authenticated web client is used.
4. Extracts audio (ffmpeg, mono 16kHz) and transcribes it with OpenAI whisper,
   writing `<date>-<abbr>_transcript.log` into `--log-dir`.

## Authentication (required)

Vimeo needs a logged-in account to download. Export your cookies from the browser
you're logged into Vimeo on:

```bash
# Option A: explicit Netscape-format cookie file
python3 download_trsd.py --cookies ~/vimeo_cookies.txt

# Option B: pull cookies directly from a running Chrome/Chromium session
python3 download_trsd.py --cookies-from-browser chrome
```

To export an explicit cookie file, use a browser extension (e.g. "Cookie-Editor")
to copy all cookies for `vimeo.com` and save them as Netscape JSON. The connected
debug browser here has **no** logged-in Vimeo account — you'll need to log in via
your own browser or export cookies from one that is.

## Usage

```bash
# Activate whisper first (required for transcription)
source openai-whisper/bin/activate

# Download + transcribe all board/committee meetings
python3 download_trsd.py --cookies ~/vimeo_cookies.txt

# Just download everything (including concerts/events), keep files, no whisper
python3 download_trsd.py --cookies ~/vimeo_cookies.txt --all --keep-video --no-transcribe

# Use a smaller/faster model
python3 download_trsd.py --cookies ~/vimeo_cookies.txt --model base
```

## Options

| Flag | Description |
|------|-------------|
| `--cookies FILE` | Netscape-format cookie jar for Vimeo login (required) |
| `--cookies-from-browser NAME` | Pull cookies from a running browser instead |
| `--out DIR` | Output directory for downloaded videos (default: `./downloads`) |
| `--log-dir DIR` | Directory for transcript logs (default: same as `--out`) |
| `--model NAME` | Whisper model (`base`, `small`, `medium`, `turbo`; default: `turbo`) |
| `--all` | Download every video, not just meetings |
| `--keep-video` | Keep downloaded videos after transcription |
| `--no-transcribe` | Skip whisper; keep the video only |

## Meeting classification

Titles are matched against these patterns (first match wins):

| Pattern | Abbreviation |
|---------|--------------|
| `budget committee` | `budget` |
| `board of selectmen`, `TRSB`, `TRS meeting` | `bos` |
| `planning board`, `planning committ` | `planning` |
| `zoning board`, `zba` | `zba` |
| `recreation`, `remuneration` | `recreation` |

Anything else (concerts, graduations, etc.) is skipped unless `--all` is passed.

<a href="https://www.buymeacoffee.com/jasonrussell" target="_blank"><img src="https://cdn.buymeacoffee.com/buttons/default-orange.png" alt="Buy Me A Coffee" height="41" width="174"></a>
