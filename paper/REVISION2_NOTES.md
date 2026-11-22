# Second revision: response to the reviews, claim-to-evidence map, and how to reproduce

Manuscript: `paper/deliverables/AeroSoar_DSP_paper_IEEE_revision2.pdf` (source `paper/paper_r2.tex`). The first revised paper (`AeroSoar_DSP_paper_IEEE_revised.*`) is unchanged.

## 1. What is new in this revision

| New element | Where in the paper | Evidence (result files in `paper/results/`) |
|---|---|---|
| Diagnosis of the Kalman speed-change deficit: true-speed-rate diagnostic (Cx), delay matching (CL), scheduled blend (CS) | Sec. IV-B, VIII-B, Table IV, Fig. 1 | `rev2_pipelines*.json` |
| Drift-rate state, attitude allowance, accelerometer-based drag term and its alignment cost | Sec. IV, VIII-C, Table V | `rev2_pipelines.json` |
| Delay law of the thermal-centre error, e^2 = e0^2 + (kV tau)^2, and delay-compensated assignment | Sec. VI, VIII-F, Fig. 3 | `rev2_closed_*.json`, `rev2_lag_*.json` |
| Detector operating points at matched false-alarm limits with paired intervals; Monte-Carlo false-alarm intervals; speed-doublet test; sensitivity of the adaptive constants | Sec. VIII-E, Table VI, Fig. 2 | `rev2_detect_*.json`, `rev2_arl.json`, `rev2_variants.json` |
| Recorded data: sensor check, noise replay in the simulator, replay of 53 min of recorded flight against a GNSS-based reference | Sec. VIII-D, Tables IV and VII | `real_alfa.json`, `rev2_*_trace.json`, `real_replay_summary.json` |

## 2. Point-by-point response

### Reviewer 1 (methods and theory)
1. *Eq. (1): sign convention and neglected terms.* Stated in Sec. III-B: w positive upward, quasi-steady, small flight-path angle, airspeed in the kinetic term, horizontal-wind acceleration and wind-gradient terms neglected (also absent from the simulator).
2. *Delay formulas and validation inconsistent.* Closed-form delays are given once (Sec. IV-A); measured delays appear only in the filter table (Table III) and are labelled as such. The "delay-free RMSE" column was removed.
3. *Accelerometer projection and attitude error.* The simulator now has an attitude error (Gauss-Markov, 1 degree) that leaks into the vertical acceleration as -g tan(phi) dphi. An allowance on the accelerometer-bias state is tested; a white allowance is shown not to help (Sec. IV-B, VIII-C).
4. *Kalman deficit attributed to an untested hypothesis.* Tested: replacing the speed-rate term by the true value lowers the speed-change error from 0.70 to 0.05 m/s; delay matching restores 0.26 m/s; a combined energy-rate Kalman filter and a complementary filter are included (Table IV).
5. *Barometer drift is outside the model class.* A drift-rate state (observability rank 5 with GNSS altitude, verified numerically) removes the bias (+0.09 to +0.02 m/s).
6. *Filter-comparison fairness, 0.4 Hz optimum.* An asymmetric Savitzky-Golay filter (look-ahead = accepted delay) and lead compensation were added; at equal delay the Butterworth filter has no advantage. The closed-loop optimum is shown to differ from the open-loop one, with paired tests (Sec. VIII-F). The dependence of the optimum on the simulated thermal scale and noise is stated as a limitation.
7. *CUSUM theory.* Monte-Carlo false-alarm intervals on autocorrelated input are given (an AR(1) process reproduces the fixed CUSUM, i.i.d. input does not); the adaptive constants were varied one at a time (43-63 % detection at 0.9 m/s turbulence), so they are reported as hand-set and not optimal.
8. *No theory linking latency and centring.* Sec. VI derives the relation, Sec. VIII-F fits it (k = 1.15-1.17, R^2 = 0.99-1.00 in calm air and tailwind) and uses it to derive a compensation that removes the delay penalty.

Questions: (1) the along-track and vertical accelerations are rotated with the attitude in the simulator, and the attitude error enters as above; (2) a combined energy-rate filter (D, D2) is now included, and the along-track accelerometer cannot supply the speed rate (identity shown in Sec. IV-C); (3) yes, the drift-rate state removes the ramp; (4) the 0.4 Hz optimum is specific to the simulated ranges and to open-loop RMSE; (5) the adaptive constants were hand-set, their sensitivity is now reported.

### Reviewer 2 (experiments and practical impact)
1. *No real data; shared assumptions.* Added: comparison of the sensor assumptions with the ALFA flights, replay of the recorded barometer and accelerometer noise in the simulator, noise levels unknown to the filters (x2, x3), and a replay of 53 min of recorded flight through the pipelines against an independent GNSS-based reference (Sec. VIII-D). What remains missing: recorded flights with thermals and a true w.
2. *Closed-loop claims not statistically supported.* Paired t-tests with Holm adjustment against the default pipeline; comparisons of cutoffs with Bonferroni control over the 10 comparisons made (Sec. VIII-F). The statement about centre error now rests on tests, not on the extreme cutoff.
3. *Non-monotonic closed-loop results.* The GNSS-delay and pitot-lag sweeps were dropped from the paper (they were within sampling noise); the closed-loop section reports the cutoff and pipeline comparisons that are significant.
4. *Detector evaluation.* Operating points at two matched false-alarm limits with paired bootstrap intervals; the adaptive detector is shown to be worse in smooth air and better in moderate and strong turbulence; a speed-doublet test checks detection during manoeuvres; Table VI is relabelled.
5. *Different flight sets, inconsistent numbers.* One flight set (30 guided flights) is used for every pipeline table; the straight-flight set is used only for the filter comparison and is named as such (Sec. VII).
6. *Baselines.* Added: asymmetric Savitzky-Golay, lead compensation, complementary filter, energy-rate Kalman filter.
7. *Speed-change stratum thin.* Intervals are given for all strata; the small number of events per flight is stated in the limitations.
8. *Statistics.* Student-t intervals (2.045 for n = 30), paired differences, Holm and Bonferroni control.
9. *Practical relevance.* Mission speed is insensitive to most choices (stated); the centre error and the false-alarm behaviour are not.

### Reviewer 3 (clarity and positioning)
1. *Claims outrun evidence.* The abstract and conclusion were rewritten; detector comparisons are at matched false-alarm limits.
2. *Positioning.* Related work now names the asymmetric Savitzky-Golay estimator (Daugherty and Langelaan, 2014), Edwards and Silverberg (2010) and total-energy-compensated climb rate from trajectories (Gross et al., 2019). The gust-compensated variometers of Ellis (2013) and the motor-glider netto variometer of Tin (2022) named in the reviews could not be found in Crossref and are not cited.
3. *Table and notation problems.* All tables are regenerated from the result files by `paper/make_tables_r2.py` and `make_tables_r2b.py`; metrics (thermal-centre error, detection ground truth, true w) are defined in Sec. VII.
4. *Claim-to-evidence mapping.* Section 1 above.
5. *Generalisation.* One aircraft, one polar, Gaussian thermals: stated as limitations.

### Not done
Real thermal flights with a true w; thermal profiles beyond Gaussian; comparison with the gust-compensated variometers named by the reviewers; an attitude-error state in the Kalman filter; a working matched-filter detector (the experiment is in the repository and failed to reach useful detection).

## 3. Reproduction

Run from `gui/` (Node.js 18 or later). Seeds are fixed in the scripts. On a 16-core machine each mode takes 0.5 to 15 minutes; the adverse-condition missions are the slowest.

```
node tests/paper_rev2.mjs tune ; node tests/paper_rev2.mjs tuneS      # tuning on seeds 500-529
node tests/paper_rev2.mjs pipelines                                    # also REAL=1 and TRACE=1
node tests/paper_rev2.mjs detect kf 0                                  # detect <source> <speed doublet>; MATCHED=1 adds the matched filter
node tests/paper_rev2.mjs variants ; node tests/paper_rev2.mjs arl
node tests/paper_rev2.mjs closed pf tailwind ; node tests/paper_rev2.mjs closed pf tailwind kf5s
node tests/paper_rev2.mjs lag pf tailwind ; node tests/paper_rev2.mjs artefact
```

Recorded data (needs `dataflash.zip` of the ALFA dataset, CC0, extracted to `data/alfa/dataflash/`):

```
uv run --with pymavlink --with numpy python -I paper/make_noise_traces.py     # noise traces for TRACE=1
uv run --with pymavlink --with numpy python -I paper/export_alfa_streams.py   # airborne segments
node tests/real_replay.mjs                                                    # from gui/
python paper/analyze_real_replay.py
```

Analysis, tables and figures: `python paper/analyze_rev2.py`, `python paper/make_tables_r2.py`, `python paper/make_tables_r2b.py`, `python paper/make_figures_r2.py`.
