from __future__ import annotations

import numpy as np
import soundfile as sf
from pathlib import Path
from typing import List, Tuple

def load_track(path: Path, target_sr: int | None = None) -> Tuple[np.ndarray, int]:
    """Read audio, convert to mono, optional resample."""
    data, sr = sf.read(path, always_2d=False)
    if data.ndim > 1:
        data = np.mean(data, axis=1)
    if target_sr and sr != target_sr:
        from scipy.signal import resample_poly
        g = np.gcd(sr, target_sr)
        data = resample_poly(data, target_sr // g, sr // g)
        sr = target_sr
    return data.astype(np.float32), sr


def rms(signal: np.ndarray) -> float:
    return float(np.sqrt(np.mean(signal ** 2)))


def energy(signal: np.ndarray) -> float:
    return float(np.sum(signal ** 2))


def crest_factor(signal: np.ndarray) -> float:
    pk = np.max(np.abs(signal))
    return float(pk / rms(signal)) if pk else 0.0


def correlation(x: np.ndarray, y: np.ndarray) -> float:
    n = min(len(x), len(y))
    if n == 0:
        return 0.0
    x = x[:n] - np.mean(x[:n])
    y = y[:n] - np.mean(y[:n])
    denom = np.sqrt(np.sum(x ** 2) * np.sum(y ** 2))
    return float(np.sum(x * y) / denom) if denom else 0.0


def auto_gain_staging(tracks: List[np.ndarray], target_rms_dbfs: float = -20.0) -> Tuple[List[np.ndarray], List[float]]:
    """Global gain so that summed RMS == target."""
    target_lin = 10 ** (target_rms_dbfs / 20)
    power_sum = np.sum([rms(t) ** 2 for t in tracks])
    g = target_lin / np.sqrt(power_sum)
    scaled = [t * g for t in tracks]
    gains_db = [20 * np.log10(g)] * len(tracks)
    return scaled, gains_db

def unique_path(p: Path) -> Path:
    if not p.exists():
        return p
    c = 1
    while True:
        new = p.with_name(f"{p.stem}_{c}{p.suffix}")
        if not new.exists():
            return new
        c += 1


def save_file(obj: np.ndarray | str, path: Path, sr: int | None, mode: str):
    if mode == "increment":
        path = unique_path(path)
    if isinstance(obj, np.ndarray):
        sf.write(path, obj.astype(np.float32), sr)
    else:
        path.write_text(obj, encoding="utf-8")
    return path

def analyse_project(folder: Path,
                    target_sr: int = 48_000,
                    ref_pair: Tuple[int, int] = (0, 1),
                    target_rms_dbfs: float = -20.0,
                    save_mode: str = "overwrite"):
    wavs = sorted(folder.glob("*.wav"))
    if len(wavs) < 2:
        raise ValueError("Mindestens zwei WAV-Dateien erforderlich.")

    print("Gefundene Tracks:")
    for i, f in enumerate(wavs):
        print(f"  [{i}] {f.name}")

    tracks, sr = [], None
    for f in wavs:
        x, sr = load_track(f, target_sr)
        tracks.append(x)

    min_len = min(len(t) for t in tracks)
    tracks = [t[:min_len] for t in tracks]

    energies = [energy(t) for t in tracks]
    rms_vals = [rms(t) for t in tracks]
    crest_vals = [crest_factor(t) for t in tracks]

    i, j = ref_pair
    rho = correlation(tracks[i], tracks[j])

    print("\nEinzelspur-Analyse:")
    for idx, (E, r, c) in enumerate(zip(energies, rms_vals, crest_vals)):
        print(f"Track {idx}: Energie={E:.2e}, RMS={20*np.log10(r):.2f} dBFS, C={c:.2f}")
    print(f"\nKorrelationsfaktor Track {i}/{j}: {rho:.3f}")

    raw_mix = np.sum(np.stack(tracks), axis=0)
    raw_rms = rms(raw_mix)
    print(f"\nRMS Roh-Summe: {20*np.log10(raw_rms):.2f} dBFS")

    scaled, gains_db = auto_gain_staging(tracks, target_rms_dbfs)
    mix = np.sum(np.stack(scaled), axis=0)
    mix_rms = rms(mix)
    print(f"Nach Gain-Staging: {20*np.log10(mix_rms):.2f} dBFS (Ziel {target_rms_dbfs} dBFS)")

    out_dir = folder / "Output"
    rep_dir = folder / "Output"
    out_dir.mkdir(exist_ok=True)
    rep_dir.mkdir(exist_ok=True)

    for f, s, g in zip(wavs, scaled, gains_db):
        p = save_file(s, out_dir / f"{f.stem}_scaled.wav", sr, save_mode)
        print(f"↳ {p.name}  (Gain {g:+.2f} dB)")

    save_file(mix, out_dir / "mix_sum.wav", sr, save_mode)

    report = []
    for idx, (E, r, c, g) in enumerate(zip(energies, rms_vals, crest_vals, gains_db)):
        report.append(f"Track {idx}: Energie={E:.2e}, RMS={20*np.log10(r):.2f} dBFS, C={c:.2f}, Gain={g:+.2f} dB\n")
    report.extend([
        f"\nKorrelationsfaktor {i}/{j}: {rho:.3f}\n",
        f"RMS Roh-Summe: {20*np.log10(raw_rms):.2f} dBFS\n",
        f"RMS nach Pegeln: {20*np.log10(mix_rms):.2f} dBFS (Ziel {target_rms_dbfs} dBFS)\n",
    ])

    rpt_path = save_file("".join(report), rep_dir / "pegel_report.txt", None, save_mode)
    print(f"Report: {rpt_path}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="PA3 – Analyse & Pegeln")
    parser.add_argument("project_dir", type=Path, help="Ordner mit WAV‑Stems")
    parser.add_argument("--sr", type=int, default=48_000,help="Samplerate für Verarbeitung [Hz]")
    parser.add_argument("--target", type=float, default=-20.0,help="Ziel‑RMS in dBFS für die Summe")
    parser.add_argument("--pair", type=int, nargs=2, default=(0, 1),help="Index‑Paar für Korrelationsanalyse")
    args = parser.parse_args()

    analyse_project(args.project_dir, args.sr, tuple(args.pair), args.target)
