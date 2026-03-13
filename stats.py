import os
import pickle
import numpy as np

STATS_PATH = "models\\mean_std.pkl"

def calculate_stats_from_record(song):
    total_sum, squared_sum, pixels = 0, 0, 0

    for timing in song:
        img = song[timing]["image"].astype("float") / 255.0

        total_sum += img.sum()
        squared_sum += (img**2).sum()
        pixels += img.size

    mean = total_sum / pixels
    variance = squared_sum / pixels - mean ** 2
    M2 = variance * pixels
    return pixels, mean, M2

def combine_stats(stats, save_to_file):
    pixels = sum(p for p, _, _ in stats)
    mean = sum(p*m for p, m, _ in stats) / pixels
    M2 = sum(M2 + p * (m - mean) ** 2 for p, m, M2 in stats)

    std = np.sqrt(M2 / pixels)

    if save_to_file: save_stats({"pixels": pixels, "mean": mean, "M2": M2})
    return mean, std

def save_stats(stats):
    with open(STATS_PATH, "wb") as f:
        pickle.dump(stats, f)

def load_current_stats():
    if os.path.exists(STATS_PATH):
        with open(STATS_PATH, "rb") as f:
            stats = pickle.load(f)

        pixels = stats["pixels"]
        mean = stats["mean"]
        M2 = stats["M2"]

        return pixels, mean, M2
    else:
        return None

def calculate_current_mean_std():
    stats = load_current_stats()
    if stats is not None:
        pixels, mean, M2 = stats
        std = np.sqrt(M2 / pixels)
        return mean, std
    else:
        return None, None

def update_global_stats(song):
    new_pixels, new_mean, new_M2 = calculate_stats_from_record(song)

    stats = load_current_stats()
    if stats is not None:
        old_pixels, old_mean, old_M2 = stats

        pixels = old_pixels + new_pixels
        delta_mean = new_mean - old_mean
        mean = old_mean + delta_mean * (new_pixels / pixels)
        M2 = old_M2 + new_M2 + delta_mean**2 * (old_pixels * new_pixels / pixels)
    else:
        pixels = new_pixels
        mean = new_mean
        M2 = new_M2

    with open(STATS_PATH, "wb") as f:
        pickle.dump({
            "pixels": pixels,
            "mean": mean,
            "M2": M2
        }, f)

    return pixels, mean, M2