# AeroSoar

Simulation environment and experiment code for the study *Causal Signal Processing for Vertical Air Velocity Estimation and Thermal Detection in Autonomous Soaring Gliders: A Simulation Study*. The project is simulation-first: all results are obtained with assumed aircraft, atmosphere and sensor models, and no claim of flight readiness is made. No flight data are included. Recorded flights of a powered fixed-wing aircraft (the public ALFA dataset, CC0) are used only to check the sensor assumptions; see below.

## Contents

| Path | Contents |
|---|---|
| `gui/js/sim/` | Simulation core: glider and atmosphere (`physics.js`), sensors (`sensors.js`), causal filters and Welch PSD (`dsp.js`), vertical-air-velocity pipelines, CUSUM, Kalman, extended Kalman and particle filters (`estimation.js`), guidance (`guidance.js`), integration (`sim.js`) |
| `gui/` | Browser interface for the simulator (three.js is bundled in `gui/vendor`) |
| `gui/tests/` | Headless checks and the experiment scripts used for the paper |
| `paper/` | Analysis and figure scripts, the raw result files (`paper/results/*.json`) and the four figures of the paper |
| `ml/`, `gui/ml/` | Training script and trained weights of the neural estimators (not used in the signal-processing study) |
| `dsp/dsp_demo.py` | Stand-alone demonstration of the filters |

## Requirements

Node.js 18 or later (for the simulator and experiments) and Python 3.10 or later with `numpy`, `scipy` and `matplotlib` (for analysis and figures). No build step is needed.

## Running the simulator

**Live demonstration (no installation):** https://aishadin330-glitch.github.io/Aerosoar/

```bash
python -m http.server 8010 --directory gui --bind 127.0.0.1
```

Open the address `http://localhost:8010` in a browser (it exists only on your own computer while this command is running) and select **Start**. A quick check without a browser:

```bash
node gui/tests/headless.mjs 45
```

## Reproducing the results

All experiment scripts are run from the `gui` directory and write into `paper/results/`. Seeds are fixed inside the scripts (30 flights per condition use seeds 100 to 129; detector tuning uses seeds 500 to 529 and the test uses seeds 1000 to 1119), so the numbers are reproducible.

| Paper section | Command (from `gui/`) | Output |
|---|---|---|
| Filters, detectors, ablation, spectra | `node tests/paper_dsp.mjs` | `dsp.json` |
| Pipelines under faults | `node tests/paper_experiments.mjs pipelines` | `pipelines.json` |
| Mission comparison | `node tests/paper_experiments.mjs missions calm` (also `tailwind`, `adverse`) | `missions_*.json` |
| Closed-loop latency | `node tests/paper_rev.mjs latency fc` (also `gnss`, `pitot`) | `rev_latency_*.json` |
| Detector operating curves | `node tests/paper_rev.mjs roc` | `rev_roc.json` |
| Kalman sensitivity | `node tests/paper_rev.mjs kalman on` (also `off`) | `rev_kalman_*.json` |
| Stress tests | `node tests/paper_rev.mjs stress` | `rev_stress.json` |
| Offline filter baselines | `node tests/paper_rev.mjs signals` | `rev_signals.json` |

Then, from the repository root:

```bash
python paper/analyze_rev.py        # tables of the revision experiments -> paper/results/rev_summary.json
python paper/make_dsp_results.py   # tables from dsp.json
python paper/make_figures.py       # figures -> paper/figures/
```

The mission scripts also evaluate the learned estimator and therefore read `gui/ml/transformer.json`, which is included.

## Source organisation of the simulator

Ground truth (`physics.js`) is kept separate from everything the estimators see: the estimators in `estimation.js` operate on sensor measurements only, each carrying its acquisition time.

## Limitations

The JavaScript implementation is a prototype. Sensor noise is white or first-order coloured Gaussian, the thermal model is a Gaussian updraft, and all parameters are assumed values listed in the paper. Validation on recorded flight data has not been done.

## Licence

Code is released under the MIT licence (see `LICENSE`). The bundled three.js library in `gui/vendor` keeps its own licence.

## Citation

If you use this code, please cite the paper and the archived release (DOI to be added).

## Check against recorded flights

Download `dataflash.zip` from the ALFA dataset (https://kilthub.cmu.edu/articles/dataset/ALFA_A_Dataset_for_UAV_Fault_and_Anomaly_Detection/12707963, licence CC0), extract it to `data/alfa/dataflash/`, and run

```bash
uv run --with pymavlink --with scipy --with numpy python -I paper/real_data_check.py
```

The medians over the logs are written to `paper/results/real_alfa.json` (the file included here is the result used in the paper).
