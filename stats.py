import os
import pickle
import numpy as np

STATS_PATH = "models\\mean_std.pkl"

def calculate_stats_from_song(song):
    total_sum, squared_sum, pixels = 0, 0, 0

    for img in song:
        img = img["image"].astype("float") / 255.0

        total_sum += img.sum()
        squared_sum += (img**2).sum()
        pixels += img.size

    mean = total_sum / pixels
    variance = squared_sum / pixels - mean ** 2
    M2 = variance * pixels
    return pixels, mean, M2

def load_current_stats():
    if os.path.exists(STATS_PATH):
        with open("models\\mean_std.pkl", "rb") as f:
            stats = pickle.load(f)

        pixels = stats["n"]
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
        return None

def update_global_stats(song):
    new_pixels, new_mean, new_M2 = calculate_stats_from_song(song)

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