# DeepLISA

DeepLISA is an AI-assisted, physics-constrained image-analysis workflow for fluorescence microchamber readout. It converts fluorescence-positive chamber events into auditable positive counts, Poisson occupancy, and calibration-based concentration estimates. The workflow was developed for dPAC-DeepLISA microchamber analysis and can also serve as a reference framework for related digital immunoassay or digital CRISPR image-readout experiments.

DeepLISA is research software. It is not a clinical diagnostic tool and does not provide clinical decision support.

## Key Features

- Single-image and folder-based fluorescence microchamber analysis.
- Top-hat background correction, CLAHE local contrast enhancement, Gaussian smoothing, and adaptive thresholding.
- Spot-first positive-candidate proposal using connected components, distance-transform separation, and contour extraction.
- Candidate-level evidence scoring based on local signal-to-background ratio, local z-score, support fraction, area, circularity, and radius consistency.
- Optional grid-consistency checks for chamber-array images.
- PSF-inspired signal-consistency verification and image-level quality control.
- Output of positive chamber count K, Poisson occupancy lambda, estimated concentration, overlay images, candidate tables, and batch-level QC reports.
- GUI-based exploratory use and command-line candidate export.

## Scope

DeepLISA is intended for fluorescence micrographs of digital chamber arrays where positive signals appear as local circular or near-circular fluorescent spots and chamber geometry is reasonably stable.

For local crops, severely defocused images, bubbles, contamination, saturated bright artifacts, black borders, incomplete chip regions, or strongly nonuniform illumination, the workflow may still produce candidate counts. Such outputs should be interpreted together with overlay images and quality-control fields and should not be used as formal quantitative results without inspection.

## Requirements

Recommended environment:

- Python 3.9 or later
- Windows 10/11, macOS, or Linux

Install dependencies:

```bash
pip install -r requirements.txt
```

Main dependencies:

- `opencv-python`
- `numpy`
- `scipy`
- `Pillow`
- `tkinterdnd2`
- `python-dotenv`
- `requests`

`requests` and `python-dotenv` are only required for optional external parameter-suggestion helpers. The default image-analysis workflow does not require private API access.

## Quick Start

### Launch the GUI

Run the following command from the code directory:

```bash
python main.py
```

The GUI supports:

- single-image import
- folder-based batch processing
- chamber radius, chamber number, chamber volume, and molecular-weight settings
- local ROI mode
- result overlay inspection and analysis logs

### Export candidate tables from the command line

Export candidate records for a single image or a folder:

```bash
python export_candidate_table.py <image-or-folder> --out results/candidate_export
```

Run with local deterministic rules and without the optional external helper:

```bash
python export_candidate_table.py <image-or-folder> --out results/candidate_export --no-ai
```

Analyze local ROI crops:

```bash
python export_candidate_table.py <image-or-folder> --out results/local_roi_export --local-roi
```

Common parameters:

```bash
--radius 7
--mw 4.3
--volume-pl 785
--n-total 21600
```

`n-total` is the theoretical chamber number for one detection region by default. Local ROI crops are generally suitable for candidate inspection and local counting only, not for formal concentration inference.

## Outputs

Batch processing or candidate export can generate:

- `result_<filename>.jpg`: overlay image with accepted positive candidates.
- `batch_report.csv`: image-level count, concentration estimate, parameters, confidence, and QC metrics.
- `candidate_table.csv`: accepted positive candidate records.
- `all_candidate_table.csv`: candidate-level records including accepted, weak, uncertain, and artifact categories.
- `*_convergence.json`: iteration, count, parameter, and quality-metric logs.
- `spatial_quadrat_counts.csv` and `spatial_quadrat_residuals.csv`: spatial QC outputs when available.

These files are intended to make image-derived counts traceable. Formal quantitative use should check count values, overlay images, and QC fields together.

## File Overview

```text
main.py                         GUI entry point
sample_analysis.py              single-image and batch orchestration
processing_pipeline.py           core image-processing and candidate-detection workflow
advanced_filtering.py            image enhancement and filtering utilities
image_utils.py                   image loading, saving, and overlay drawing
concentration_calculator.py      Poisson occupancy and concentration conversion
quality_control.py               basic quality-control utilities
psf_verification.py              PSF-inspired signal-consistency checks
grid_detection.py                grid-assisted consistency and array-geometry tools
region_inference.py              single-region and multi-region scope inference
spatial_qc.py                    spatial distribution and chamber-opportunity weighted QC
design_prior.py                  chip-design prior loading and organization
agent_core.py                    iterative parameter adjustment and workflow control
agent_quality.py                 composite confidence and quality scoring
agent_tools.py                   analysis-tool wrappers
agent_batch_context.py           batch-context consistency logic
export_candidate_table.py        command-line candidate and audit-table export
config.py                        default parameters and environment configuration
```

## Environment Variables

DeepLISA can read selected defaults from environment variables:

```text
RESULT_DIR
TEMP_DIR
RADIUS
DEFAULT_MW_KDA
DEFAULT_VOL_PL
DEFAULT_TOTAL_CHAMBERS
LANGUAGE
AUTO_REGION_INFERENCE
DETECTION_MODE
EXPORT_DETECTION_AUDIT
EXPORT_GRID_CANDIDATES
```

Optional external parameter-suggestion helpers may read:

```text
GLM_API_KEY
GLM_BASE_URL
AI_MODEL
```

Do not upload `.env` files or private API keys to a public repository. If example configuration is needed, provide `.env.example` without real credentials.

## Public Release Checklist

Recommended files to include:

```text
README_CN.md
README_EN.md
requirements.txt
.gitignore
LICENSE
examples/README.md
source code files
```

Recommended files and folders to exclude:

```text
.env
.venv/
.idea/
__pycache__/
results/
tmp/
raw microscopy images
batch output folders
internal repair notes
logs containing local paths or private information
```

## Notes

- DeepLISA outputs image-analysis results and should be interpreted together with assay design, calibration data, and QC metrics.
- Overlay images should be inspected for representative samples, especially in low signal-to-noise, high-background, or local ROI cases.
- The optional external parameter-suggestion branch is not required to reproduce the core image-analysis workflow.
- For publications or public datasets, report the software version, parameter settings, candidate tables, and batch reports to support independent review.

## Citation

If you use DeepLISA or its image-analysis workflow, please cite the associated dPAC-DeepLISA manuscript or this code repository. DOI and publication details can be added here after publication.

## License

Please add a clear `LICENSE` file before public distribution. Without an explicit license, no permission is granted for third-party copying, modification, or redistribution.
