# osu! AI Cursor Player
An experimental computer vision project for controlling the cursor in osu! using deep learning and real-time screen capture.

The project records gameplay data, synchronizes it with beatmap timings, trains a neural network on gameplay frames, and then predicts cursor movement during play.


# Features
- Real-time osu! screen capture
- Beatmap parsing from `.osu` files
- Automatic gameplay dataset generation
- Training pipeline built with PyTorch
- Real-time cursor prediction
- Slider support
- Dataset normalization and statistics caching


# How It Works

## Beatmap Processing
The project parses `.osu` beatmaps and extracts object timings and positions.

These timings are converted into cursor coordinates relative to the game window.


## Gameplay Recording
During gameplay:

- frames are captured directly from the osu! window
- gameplay timings are synchronized with beatmap object positions
- image-position pairs are stored for training

Recorded data is saved into `.pkl` files.


## Training
The neural network is trained on short sequences of gameplay frames and learns to predict cursor coordinates.

The training pipeline includes:

- dataset normalization
- batching with PyTorch DataLoader
- regression-based position prediction
- model checkpoint saving


## Real-Time Playing
In play mode the application:

1. Captures new gameplay frames
2. Preprocesses them
3. Runs inference with the trained model
4. Moves the mouse cursor in real time


# Installation
```bash
pip install torch torchvision
pip install pyqt6
pip install opencv-python
pip install dxcam
pip install pygetwindow
pip install mouse
pip install pywin32
pip install osuparser
```


# Usage

## Select Mode
```python
RECORD = 0
TRAIN = 1
PLAY = 2
```


## Example
```python
window = Player(
    ["Gira Gira", "Rory", "Mirror"],
    model_name="0.95 - best",
    img_size=(120, 60),
    mode=PLAY
)
```


# Recording Dataset
Use `RECORD` mode to capture gameplay data and generate training samples.

```python
mode=RECORD
```


# Training Model
Use `TRAIN` mode to train the neural network on recorded gameplay.

```python
mode=TRAIN
```


# Playing
Use `PLAY` mode to run the trained model in real time.

```python
mode=PLAY
```


# Notes
- The project is currently Windows-only
- Spinner objects are not implemented yet
- The project focuses only on cursor movement prediction
- Mouse clicks are not automated


# Project Goal
The main goal of the project is to experiment with:

- computer vision for rhythm games
- gameplay imitation learning
- real-time neural network inference
- automated dataset generation from games