from __future__ import annotations
import numpy as np
import soundfile as sf
from pathlib import Path
from typing import List, Tuple, Dict

def load_track(path: Path, target_sr: int | None = None) -> Tuple[np.ndarray, int]:
    """Lädt eine Audiospur als Mono‑Signal.

    Parameters
    ----------
    path : Path
        Pfad zur WAV/AIFF/FLAC‑Datei.
    target_sr : int | None
        Wenn angegeben, wird bei Bedarf auf diese Samplerate resampelt.

    Returns
    -------
    data : np.ndarray
        1‑D float32‑Signal im Bereich [‑1, 1].
    sr : int
        Verwendete Samplerate (Hz).
    """
    data, sr = sf.read(path, always_2d=False)
    if data.ndim > 1:
        # in Mono zusammenfassen
        data = np.mean(data, axis=1)
    if target_sr and sr != target_sr:
        from scipy.signal import resample_poly
        gcd = np.gcd(sr, target_sr)
        data = resample_poly(data, target_sr // gcd, sr // gcd)
        sr = target_sr
    return data.astype(np.float32), sr


def energy(signal: np.ndarray) -> float:
    """Totale Energie des Signals (Summe x^2)."""
    return float(np.sum(np.square(signal)))


def rms(signal: np.ndarray) -> float:
    """Quadratischer Mittelwert (Effektivwert)."""
    return float(np.sqrt(np.mean(np.square(signal))))


def crest_factor(signal: np.ndarray) -> float:
    """Scheitelfaktor C = x̂ / x_eff."""
    peak = np.max(np.abs(signal))
    return float(peak / rms(signal)) if peak else 0.0


def correlation(x: np.ndarray, y: np.ndarray) -> float:
    """Korrelationsfaktor (Pearson) zweier gleich langer Signale."""
    n = min(len(x), len(y))
    if n == 0:
        return 0.0
    x = x[:n] - np.mean(x[:n])
    y = y[:n] - np.mean(y[:n])
    denom = np.sqrt(np.sum(x**2) * np.sum(y**2))
    return float(np.sum(x * y) / denom) if denom else 0.0

def auto_gain_staging(tracks: List[np.ndarray], target_rms_dbfs: float = -20.0) -> Tuple[List[np.ndarray], List[float]]:
    """Skaliert n Spuren so, dass die Summe ungefähr den Ziel‑RMS erreicht.

    Annahme: Spuren sind weitgehend unkorreliert. Deshalb addieren sich die
    Leistungen: P_sum ≈ Σ P_i.  Daraus ergibt sich ein gemeinsamer Gain‑Faktor
    g , falls alle Spuren gleich gewichtet werden, oder individuelle Faktoren
    g_i, wenn ihre RMS unterschiedlich sind.

    Parameters
    ----------
    tracks : List[np.ndarray]
        Rohsignale.
    target_rms_dbfs : float
        Gewünschter RMS‑Pegelmesswert der Summe in dBFS.

    Returns
    -------
    scaled_tracks : List[np.ndarray]
        Skalierten Signale.
    gains_db : List[float]
        Entspr. Gain in dB, der auf jede Spur angewandt wurde.
    """
    target_lin = 10 ** (target_rms_dbfs / 20)
    rms_values = np.array([rms(t) for t in tracks])
    # Leistungs‑basierter globaler Gain, falls alle Spuren gleich gewichtet
    power_sum = np.sum(rms_values ** 2)
    global_gain = target_lin / np.sqrt(power_sum)

    scaled_tracks = []
    gains_db = []
    for r, t in zip(rms_values, tracks):
        # optional: gleiche Loudness pro Spur (hier: global_gain für alle)
        g = global_gain
        scaled_tracks.append(t * g)
        gains_db.append(20 * np.log10(g))
    return scaled_tracks, gains_db


# ------------------------------------------------------------
# Hauptprogramm
# ------------------------------------------------------------

def analyse_project(folder: Path, target_sr: int = 48000, ref_pair: Tuple[int, int] = (0, 1), target_rms_dbfs: float = -20.0):
    """Gesamtablauf für Praxisaufgabe 3."""
    files = sorted(p for p in folder.glob("*.wav"))
    if len(files) < 2:
        raise ValueError("Mindestens zwei WAV‑Dateien erforderlich.")

    print(f"Gefundene Tracks ({len(files)}):")
    for i, f in enumerate(files):
        print(f"  [{i}] {f.name}")

    tracks, sr = [], None
    for f in files:
        x, sr = load_track(f, target_sr)
        tracks.append(x)

    energies = [energy(t) for t in tracks]
    rms_vals = [rms(t) for t in tracks]
    crest = [crest_factor(t) for t in tracks]

    i, j = ref_pair
    rho = correlation(tracks[i], tracks[j])

    print("\nEinzelspur‑Analyse:")
    for idx, (E, r, c) in enumerate(zip(energies, rms_vals, crest)):
        print(f"Track {idx}: Energie={E:.2e}, RMS={20*np.log10(r):.2f} dBFS, C={c:.2f}")

    print(f"\nKorrelationsfaktor zwischen Track {i} und Track {j}: ρ = {rho:.3f}")

    # Automatisches Gain‑Staging
    scaled, gains_db = auto_gain_staging(tracks, target_rms_dbfs)
    mix = np.zeros_like(scaled[0])
    for t in scaled:
        mix[:len(t)] += t[:len(mix)]

    mix_rms = rms(mix)
    print(f"\nNach Gain‑Staging: RMS(Stems) = {20*np.log10(mix_rms):.2f} dBFS (Ziel {target_rms_dbfs} dBFS)")

    # Speichern
    out_folder = folder.parent / "Output"  # legt alles in /Output
    pegeldaten = folder.parent / "Pegeldaten"  # legt Report in /Pegeldaten
    out_folder.mkdir(exist_ok=True)
    pegeldaten.mkdir(exist_ok=True)

    for f, s, g in zip(files, scaled, gains_db):
        sf.write(out_folder / f"{f.stem}_scaled.wav", s, sr)
        print(f"↳ {f.stem}_scaled.wav  (Gain {g:+.2f} dB)")
    sf.write(out_folder / "mix_sum.wav", mix, sr)
    print("↳ mix_sum.wav  (Summensignal)")

    with open(pegeldaten / "pegel_report.txt", "w", encoding="utf-8") as fp:
        fp.write("== Einzelspur-Analyse ==\n")
        for idx, (E, r, c, g) in enumerate(zip(energies, rms_vals, crest, gains_db)):
            fp.write(f"Track {idx}: Energie={E:.2e}, RMS={20 * np.log10(r):.2f} dBFS, "
                     f"C={c:.2f}, Gain={g:+.2f} dB\n")
        fp.write(f"\nKorrelationsfaktor (Tracks {i}/{j}): ρ={rho:.3f}\n")
        fp.write(f"\nSummen-RMS nach Pegeln: {20 * np.log10(mix_rms):.2f} dBFS "
                 f"(Ziel {target_rms_dbfs} dBFS)\n")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Praxisaufgabe 3 – Analyse & automatisches Pegeln")
    parser.add_argument("project_dir", type=Path, help="Ordner mit WAV‑Stems")
    parser.add_argument("--sr", type=int, default=48000, help="Samplerate für Verarbeitung [Hz]")
    parser.add_argument("--target", type=float, default=-20.0, help="Ziel‑RMS in dBFS für die Summe")
    parser.add_argument("--pair", type=int, nargs=2, default=(0, 1), help="Index‑Paar für Korrelationsanalyse")
    args = parser.parse_args()

    analyse_project(args.project_dir, args.sr, tuple(args.pair), args.target)
