"""DSP demonstration for the AeroSoar project (same formulas as gui/js/sim/dsp.js and estimation.js).

usage: uv run --no-project --python 3.12 --with numpy --with scipy --with matplotlib python -I dsp/dsp_demo.py
Writes docs/figures/dsp_demo.png and prints a results table.
"""
import pathlib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import signal

rng = np.random.default_rng(1)
fs = 20.0                                   # barometer rate, Hz
t = np.arange(0, 120, 1 / fs)
w_true = 3.0 * np.exp(-((t - 60) ** 2) / (2 * 6.0 ** 2))   # a thermal passing: peak 3 m/s
h_true = np.cumsum(w_true) / fs + 300.0                      # altitude gained by that climb
h_baro = h_true + rng.normal(0, 0.12, t.size)                # barometer: 0.12 m noise

# 1. naive differentiation: (h[n] - h[n-1]) / dt
naive = np.r_[0, np.diff(h_baro)] * fs


# 2. filters, all causal (use only past samples)
def butter2(x, fc):
    K = np.tan(np.pi * fc / fs)             # bilinear-transform constants, as in dsp.js
    n = 1 / (1 + np.sqrt(2) * K + K * K)
    b = np.array([K * K * n, 2 * K * K * n, K * K * n])
    a = np.array([1, 2 * (K * K - 1) * n, (1 - np.sqrt(2) * K + K * K) * n])
    return signal.lfilter(b, a, x), b, a


def movavg(x, fc):
    N = max(1, round(0.443 * fs / fc))
    return np.convolve(x, np.ones(N) / N)[: x.size]


def onepole(x, fc):
    return signal.lfilter([1 - np.exp(-2 * np.pi * fc / fs)], [1, -np.exp(-2 * np.pi * fc / fs)], x)


FC = 0.4
yb, b, a = butter2(naive, FC)
sb, sa = signal.butter(2, FC, fs=fs)
assert np.allclose(b, sb) and np.allclose(a, sa), "hand-written coefficients must equal scipy's"
outs = {"Naive (no filter)": naive, "Moving average": movavg(naive, FC), "One-pole": onepole(naive, FC), "Butterworth 2nd order": yb}


def delay_s(y):                              # lag at which the filtered curve best matches the truth
    lags = np.arange(0, int(6 * fs))
    a0, a1 = int(30 * fs), int(90 * fs)      # compare the window around the thermal
    err = [np.mean((y[a0 + l: a1 + l] - w_true[a0:a1]) ** 2) for l in lags]   # output shifted back by l samples
    return lags[int(np.argmin(err))] / fs


print(f"{'Filter':<24}{'RMSE vs truth, m/s':>20}{'Delay, s':>10}")
for k, y in outs.items():
    print(f"{k:<24}{np.sqrt(np.mean((y - w_true) ** 2)):>20.2f}{delay_s(y):>10.2f}")
print("Theoretical Butterworth low-frequency group delay sqrt(2)/(2*pi*fc) =", round(np.sqrt(2) / (2 * np.pi * FC), 2), "s")

# 3. CUSUM detector on the Butterworth output (+ gust-free demo)
g, trip = 0.0, None
cus = np.zeros(t.size)
for i, v in enumerate(yb):
    g = max(0.0, g + (v - 0.5) / fs)
    cus[i] = g
    if trip is None and g > 1.5:
        trip = t[i]
print("CUSUM alarm time:", None if trip is None else round(trip, 1), "s (thermal centre at 60 s, 1 sigma = 6 s)")

# 4. figures
fig, ax = plt.subplots(2, 3, figsize=(14, 7))
ax[0, 0].plot(t, naive, color="0.6", lw=0.7, label="Naive derivative")
ax[0, 0].plot(t, w_true, "k", label="Truth")
ax[0, 0].set(title="1. Differentiating noisy altitude", xlabel="Time, s", ylabel="Climb rate, m/s")
ax[0, 0].legend()
for k, c in zip(list(outs)[1:], ["tab:orange", "tab:green", "tab:blue"]):
    ax[0, 1].plot(t, outs[k], color=c, label=k)
ax[0, 1].plot(t, w_true, "k", label="Truth")
ax[0, 1].set(title="2. Causal low-pass filters (cutoff 0.4 Hz)", xlabel="Time, s", xlim=(40, 90))
ax[0, 1].legend()
f, H = signal.freqz(b, a, worN=2048, fs=fs)
ax[0, 2].plot(f, 20 * np.log10(np.abs(H)))
ax[0, 2].axvline(FC, color="r", ls="--")
ax[0, 2].set(title="3. Butterworth frequency response", xlabel="Frequency, Hz", ylabel="Gain, dB", xlim=(0, 3), ylim=(-60, 3))
for k, c in zip(["Naive (no filter)", "Butterworth 2nd order"], ["0.5", "tab:blue"]):
    fp, P = signal.welch(outs[k], fs=fs, nperseg=256)
    ax[1, 0].semilogy(fp, P, color=c, label=k)
fw, Pw = signal.welch(w_true, fs=fs, nperseg=256)
ax[1, 0].semilogy(fw, Pw, "k", label="Truth")
ax[1, 0].set(title="4. Power spectral density (Welch)", xlabel="Frequency, Hz", ylabel="PSD", xlim=(0, 5))
ax[1, 0].legend()
gd = -np.diff(np.unwrap(np.angle(H))) / np.diff(2 * np.pi * f)
ax[1, 1].plot(f[1:], gd)
ax[1, 1].set(title="5. Butterworth group delay", xlabel="Frequency, Hz", ylabel="Delay, s", xlim=(0, 3), ylim=(0, 1.2))
ax[1, 2].plot(t, cus, label="CUSUM score")
ax[1, 2].axhline(1.5, color="r", ls="--", label="Threshold 1.5")
ax[1, 2].set(title="6. CUSUM thermal detector", xlabel="Time, s")
ax[1, 2].legend()
fig.tight_layout()
dest = pathlib.Path(__file__).resolve().parent.parent / "docs" / "figures"
dest.mkdir(parents=True, exist_ok=True)
fig.savefig(dest / "dsp_demo.png", dpi=140)
print("saved", dest / "dsp_demo.png")
