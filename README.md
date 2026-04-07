# djtool-crate-digga

*&ldquo;Zero-shot [Crate Digging](https://medium.com/cuepoint/the-lost-art-of-cratedigging-4ed652643618) &rarr; DJ Tool retrieval using Speech Activity, Music Structure and CLAP embeddings&rdquo;*

Given an audio file (or a whole library), this tool automatically finds and classifies DJ-usable segments -- acapellas, drum breaks, FX, drops, scratches, and more -- by combining:

- **[MSAF](https://github.com/urinieto/msaf)** for structural segmentation (finding intro, verse, chorus, bridge, outro boundaries)
- **[TVSM](https://github.com/biboamy/TVSM-dataset)** for speech vs music activity detection
- **[CLAP](https://huggingface.co/laion/clap-htsat-unfused)** for zero-shot audio classification against DJ tool text prompts

## WTF are DJ Tools?

In genres like Hip-Hop, RnB, Reggae/Dancehall and just about every Electronic/Dance/Club style, DJ Tools are a selection of audio files curated to heighten the DJ's musical performance and creative mixing options:

- Acapella loops
- Sound effect samples
- One-shot vocal samples
- Drum breaks and breakbeats
- Melodic hooks
- DJ Drops
- Scratch and battle loops
- And anything else to keep *ish fresh


### Crate digging, the Amen Break and a short history of the DJ tool
Before the advent of online shops peddling every kind of sonic tool, DJs would sample sections
of riffs from tracks in their vinyl libraries, triggering and looping these samples to elevate
the mix. 

Perhaps one of the most famous examples of this is the [Amen Break](https://en.wikipedia.org/wiki/Amen_break), where the drum break
in a song by the American Funk and Soul band The Winstons called “Amen, Brother”, was sampled
first by Hip-Hop producers as a tool, before it caught fire and became the basis for thousands
of songs. This very technique of sampling drum breaks became the genesis of breakbeat-centric 
genres like Hardcore, Jungle, Drum'n'Bass. The key here is that the DJ knows their music 
library inside-out and can manually excise the juiciest morsels as tools. 
 
As the amount of recorded music has continued to exponentiate since the first Amen Breaks were 
sampled, today's DJ (including the author) needs to spend a lot of dedicated time listening and 
curating their music library. To get some extra help, I propose this app to identify the 
following classes commonly found in recorded music which also function as DJ Tools. These 
sections may include acappella (vocal) intros, beat-less outros or melodic instrumental passages 
or section breakdowns with just the drum solo. 

This tool attempts to operationalize that crate-digging ethos for modern digital libraries.

## Installation

Requires **Python 3.12+**

```bash
# Clone the repo
git clone https://github.com/ruohoruotsi/djtool-crate-digging.git
cd djtool-crate-digging

# Create a virtual environment
python3.12 -m venv .venv
source .venv/bin/activate

# Install in development mode
pip install -e ".[dev]"
```

### TVSM Model Checkpoint

The TVSM speech/music detector (aka [SMAD](https://netflixtechblog.com/detecting-speech-and-music-in-audio-content-afd64e6a5bf8)) requires a model checkpoint that is not bundled with this repo. To obtain it:

1. Visit the [TVSM-dataset repo](https://github.com/biboamy/TVSM-dataset) 
2. Follow the README instructions to download the checkpoint from Google Drive
3. Place the `.pt` file in the `models/` directory

Or run:
```bash
djtool-crate-digga download-models
```

## Usage

### Full pipeline (segmentation + classification)

Process a single audio file:
```bash
djtool-crate-digga process track.wav -o output/ --format folders
```

Process an entire music library:
```bash
djtool-crate-digga process ~/Music/Library/ -o output/ --format csv
```

With pre-computed SMAD output (skip TVSM inference):
```bash
djtool-crate-digga process track.wav --smad-csv smad_output.csv -o output/
```

### Classify pre-cut segments (CLAP only)

If you already have audio segments cut, run just the classification step:
```bash
djtool-crate-digga classify segments_dir/ -o output/ --format json
```

### Options

```
djtool-crate-digga process --help

Options:
  -o, --output-dir PATH        Output directory (default: output)
  --format [folders|csv|json]  Output format (default: folders)
  --device [cpu|cuda|mps]      Torch device (default: cpu)
  --msaf-algorithm TEXT        MSAF boundary algorithm (default: sf)
  --smad-model-path PATH       Path to TVSM model checkpoint
  --smad-csv PATH              Pre-computed SMAD CSV (skip TVSM inference)
  --min-duration FLOAT         Min segment duration for classification in seconds (default: 3.0)
  --threshold FLOAT            Min confidence to keep a classification (default: 0.1)
```

### Output Formats

- **folders**: Copies classified segments into subdirectories named by class (e.g. `acapella/`, `drums/`, `drops/`)
- **csv**: Tab-separated file with source, start, end, class, confidence, and segment path
- **json**: JSON array with full classification scores for each segment

## DJ Tool Classes

The default classification classes and their CLAP text prompts:

| Class         | Prompt                                                                            |
|---------------|-----------------------------------------------------------------------------------|
| acapella      | acapella, expressively sung human vocal with background instrumental music tracks |
| instrumentals | piano, synths or strings, instrumental, guitar                                    |
| drums         | drums, a drum loop, drum solo, breakbeat, percussive elements                     |
| beatbox       | beatboxing                                                                        |
| fx            | siren, riser sound effects, whoosh, crash, synthetic, transitional effect         |
| vinyl_fx      | vinyl scratch loop, turnatablist DJ battle sounds                                 |
| drops         | a high energy, high tension, climactic, massive EDM drop                          |

## How It Works

The pipeline implements **Algorithm 1** from the [ISMIR 2024 Late-Breaking/Demo paper](legacy/lbd_paper_latex/ISMIR2024_lbd.tex) &rarr;

1. **MSAF** segments the track into structural sections (intro, verse, chorus, etc.)
2. **TVSM** detects speech vs music activity across the track
3. **Fusion** adjusts MSAF boundaries to align with speech onset/offset points, producing cleaner candidate segments
4. **torchaudio** extracts the candidate segments as audio files
5. **CLAP** classifies each segment against the DJ tool text prompts using zero-shot cosine similarity
6. Results are exported in the chosen format

## Running Tests

```bash
# Make sure dev dependencies are installed
pip install -e ".[dev]"

# Run all tests
pytest

# Run with verbose output
pytest -v

# Run with coverage
pytest --cov=djtool
```

## Project Structure

```
src/djtool/
  cli.py                  # Click CLI entry point
  config.py               # Dataclass configuration
  pipeline.py             # End-to-end orchestrator
  segmentation/
    msaf_adapter.py       # MSAF structural segmentation wrapper
    smad_adapter.py       # TVSM speech/music detection wrapper
    fusion.py             # MSAF + SMAD boundary fusion (Algorithm 1)
  classification/
    clap.py               # CLAP zero-shot classification
    prompts.py            # DJ tool class definitions and text prompts
  audio/
    io.py                 # Audio loading and segment extraction
    export.py             # Output writers (folders, CSV, JSON)
  _vendor/tvsm/           # Vendored TVSM inference code (CRNN + PCEN)
```

## Background

This project originated as an [ISMIR 2024 Late-Breaking/Demo](legacy/lbd_paper_latex/) submission exploring zero-shot crate digging for DJ tool retrieval. The original prototype scripts are preserved in [`legacy/code/`](legacy/code/).

## License

MIT
