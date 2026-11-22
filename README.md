# Care Access Gap Dashboard

A Power BI dashboard that compares health insurance coverage, poverty, chronic disease and the
location of community health centers across Virginia's 133 counties and independent cities. All of
the data is public, and a Python pipeline downloads, cleans and checks it so the results can be
reproduced.

## What is in this repository

- `data/raw/` holds the files exactly as they were downloaded. They are not stored in Git.
- `data/processed/` holds the cleaned CSV files that Power BI reads.
- `scripts/` holds the Python code for downloading, cleaning and validating the data.
- `tests/` holds the automated tests for that code.
- `powerbi/` will hold the Power BI project, saved in the .pbip format so Git can track changes.
- `docs/` holds the requirements, data dictionary, methodology, metadata log and validation report.
- `screenshots/` holds images of the finished dashboard.

## Setting up

You need Python 3.11 or later. From the project folder, create a virtual environment and install the
packages:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

The Census data requires a free API key, which you can request at
https://api.census.gov/data/key_signup.html. Save it as a Windows environment variable called
`CENSUS_API_KEY`, then open a new terminal so it takes effect. Never put the key in a file in this
repository.

```powershell
[Environment]::SetEnvironmentVariable("CENSUS_API_KEY", "your-key", "User")
```

## Running the pipeline

Run these three commands in order from the project folder. Each one prints a summary at the end and
returns exit code 0 when it succeeds, so check `$LASTEXITCODE` before moving on to the next.

```powershell
python -m scripts.download --all
python -m scripts.clean_data
python -m scripts.validate_data
```

The first command downloads all six sources into `data/raw/` and records each download in
`data/raw/download_log.csv`. The second builds the six tables in `data/processed/`. The third checks
those tables and writes the results into `docs/validation_report.md`. If a step fails, it says why
and leaves the existing files as they were.

To download a single source, use `--source` with its name, for example `--source hrsa`. The name
`acs` downloads both Census tables. Add `--force` to download again on a day when a file has already
been downloaded. If a website blocks the automated download, save the file from your browser and add
it with `--manual-file`, for example `--manual-file rucc C:\Downloads\rucc.csv`.

The cleaning step also writes `data/processed/nearest_sites.csv`, the closest health center sites to
each locality. It needs the Census population centers file, which is downloaded once with
`python -m scripts.nearest_sites`. Until that has been run, the cleaning step skips this table and
says so in its output.

To run the tests, use `python -m pytest -q`. The tests never contact the real websites.

## Site map background

The Health centers page shows every HRSA site as a dot on a scatter chart, drawn over a picture of
the localities. After the cleaning step, run this command to write two files for this into
`data/processed/`:

```powershell
python -m scripts.site_map_background
```

- `site_map_background.png`, a transparent image of the localities in light gray, drawn edge to
  edge with no margins.
- `map_bounds.csv`, one row with the longitude and latitude limits the image was drawn with, its
  aspect ratio and its size in pixels.

The image is drawn from the Census county boundary file, which `python -m scripts.download --all`
already downloads. If it is missing, download it once with
`python -m scripts.download --source boundaries`. The command above also downloads it if it has
never been downloaded. It needs `dim_geography.csv`, so run it after the cleaning step.

In Power BI, set the scatter chart's X axis (site longitude) to run from `lon_min` to `lon_max` and
its Y axis (site latitude) from `lat_min` to `lat_max`, then use the PNG as the plot area
background. The dots line up with the localities only when the axes use these exact limits.

The states drawn are set by `STATES` in `scripts/config.py`. To add a state, add its two-digit FIPS
code there, rerun the command, then import the new PNG into Power BI again and copy the new limits
from `map_bounds.csv` into the axes.

## Why the raw data is not in Git

The PLACES file alone is over 60 MB. Instead of storing the raw files, the repository keeps
`data/raw/download_log.csv`, which records where each file came from, when it was downloaded, how
many rows it had and its SHA-256 checksum. Running the download again rebuilds the same inputs, and
the checksums show whether a source has changed since.

## How the code is organised

The download code lives in `scripts/download.py`, with one small module per data source in
`scripts/sources/`. All the settings, such as URLs, release years, variable codes and column names,
are in `scripts/config.py`, so changing a release year means editing one line.

The cleaning code starts in `scripts/clean_data.py`. Each output table is built by its own module in
`scripts/cleaning/`, named after the table it produces. The validation code starts in
`scripts/validate_data.py`, with the checks themselves in `scripts/validation/`.

The data dictionary in `docs/data_dictionary.md` describes every column, and the methodology in
`docs/methodology.md` explains how each figure is produced and what it cannot tell you.

## Loading the data into Power BI

Load the six CSV files from `data/processed/`. In Power Query, change the type of every `fips`
column to Text before doing anything else. Power BI's automatic type detection would otherwise turn
codes like 51001 into numbers, and the relationships between tables would break.

## Working rules

See [CONTRIBUTING.md](CONTRIBUTING.md).

---

Prepared by Pride Mudondo.
