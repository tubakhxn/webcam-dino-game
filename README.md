# Gesture Dino

Creator/Dev: **tubakhxn**

## What is this project?
Gesture Dino recreates the Chrome T-Rex runner, but the dinosaur is driven entirely by your hand gestures. Your webcam feeds MediaPipe Hands, gestures are interpreted as commands, and Pygame renders a handcrafted ASCII 🦖 plus colorful 🌵 and 🐦 emoji obstacles.
Early on the path is clear so you can practice gestures; 🌵 cacti wait until your score passes roughly 600, and the swooping 🐦 birds only appear past ~1,600, giving you plenty of runway before things ramp up.
The latest build replaces emoji dinos with handcrafted ASCII pixel art facing left, while obstacles are now vibrant 🌵 cacti on the ground and swooping 🐦 birds overhead.

## Requirements
- Python 3.10+
- Webcam connected and available
- pip-installed dependencies:
  ```bash
  python -m pip install opencv-python mediapipe pygame
  ```

## Running locally
1. Clone or download the repo (or fork it first, see below).
2. From the project root run:
   ```bash
   python main.py
   ```
3. The first run downloads the MediaPipe hand model (~6 MB) into `models/hand_landmarker.task`.

## Gestures
- ☝️ One finger up → 🦖 jumps
- ✌️ Two fingers up → 🦕 ducks
- No hand → Dino keeps running
- 🖐️ Open palm (4+ fingers) on the Game Over screen → restart

## Forking & contributions
1. Click **Fork** on GitHub to create your copy under your account.
2. Clone your fork and point the `origin` remote to it.
3. Create a feature branch, commit changes, then open a pull request back to the original repo.
4. Describe your changes, provide gameplay GIFs if possible, and tag **tubalhx a** for review.

Feel free to remix the ASCII art, recolor the 🌵/🐦 obstacles, or add new gesture-based mechanics—just keep it keyboard-free!
