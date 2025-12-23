"""
README
======
1. Install dependencies: pip install opencv-python mediapipe pygame
2. Launch the game: python main.py
3. First launch downloads the MediaPipe hand model (~6 MB) automatically.
4. Gestures:
   - One finger raised (index) => jump
   - Two fingers raised (index + middle) => duck
   - No hand in view => default running
   - Open palm (4+ fingers) after Game Over => restart
Make sure a webcam is connected before running the program.
"""

from __future__ import annotations

import math
import random
import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
from mediapipe import Image, ImageFormat
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision as mp_vision
import pygame


MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
HAND_CONNECTIONS = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    (5, 9),
    (9, 10),
    (10, 11),
    (11, 12),
    (9, 13),
    (13, 14),
    (14, 15),
    (15, 16),
    (13, 17),
    (17, 18),
    (18, 19),
    (19, 20),
    (0, 17),
]

EMOJI_FONT_CANDIDATES = [
    "Segoe UI Emoji",
    "Segoe UI Symbol",
    "Noto Color Emoji",
    "Arial Unicode MS",
    "Apple Color Emoji",
]
_EMOJI_FONT_CACHE: Dict[int, pygame.font.Font] = {}
_EMOJI_SURFACE_CACHE: Dict[Tuple[str, int], pygame.Surface] = {}


def _get_emoji_font(size: int) -> pygame.font.Font:
    if size not in _EMOJI_FONT_CACHE:
        for name in EMOJI_FONT_CANDIDATES:
            font = pygame.font.SysFont(name, size)
            if font is not None:
                _EMOJI_FONT_CACHE[size] = font
                break
        else:
            _EMOJI_FONT_CACHE[size] = pygame.font.SysFont(None, size)
    return _EMOJI_FONT_CACHE[size]


def render_emoji_surface(char: str, size: int) -> pygame.Surface:
    key = (char, size)
    if key not in _EMOJI_SURFACE_CACHE:
        font = _get_emoji_font(size)
        surface = font.render(char, True, (0, 0, 0)).convert_alpha()
        _EMOJI_SURFACE_CACHE[key] = surface
    return _EMOJI_SURFACE_CACHE[key]


DINO_ASCII_RUN = [
    "   ######      ",
    "  #########    ",
    "  ###   ###    ",
    " ###########   ",
    " ###########   ",
    " ######  ###   ",
    " ######  ###   ",
    "     #######   ",
    "      ##  ##   ",
    "      ##  ##   ",
]

DINO_ASCII_DUCK = [
    "   ########    ",
    "  ###########  ",
    "  ###########  ",
    "  #### ######  ",
    "  #### ######  ",
    "    #######    ",
    "     #####     ",
]


def render_ascii_sprite(lines: List[str], color: Tuple[int, int, int], pixel_size: int = 6) -> pygame.Surface:
    height = len(lines)
    width = max(len(line) for line in lines)
    surface = pygame.Surface((width * pixel_size, height * pixel_size), pygame.SRCALPHA)
    for y, line in enumerate(lines):
        for x, char in enumerate(line):
            if char != " ":
                rect = pygame.Rect(x * pixel_size, y * pixel_size, pixel_size, pixel_size)
                surface.fill(color, rect)
    return surface


class GestureDetector:
    """Handles webcam capture and MediaPipe-based gesture classification."""

    def __init__(self, camera_index: int = 0, width: int = 640, height: int = 480):
        self.cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            raise RuntimeError("Unable to access webcam. Ensure it is connected and free.")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.model_path = self._ensure_model_file()
        base_options = mp_tasks.BaseOptions(model_asset_path=self.model_path)
        options = mp_vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=1,
            min_hand_detection_confidence=0.6,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            running_mode=mp_vision.RunningMode.VIDEO,
        )
        self.landmarker = mp_vision.HandLandmarker.create_from_options(options)
        self.frame_size = (width, height)
        self.timestamp_ms = 0

    def process_frame(self):
        """Returns (pygame_surface, action, finger_count, hand_present)."""
        action = "run"
        finger_count = 0
        hand_present = False
        ret, frame = self.cap.read()
        if not ret:
            blank_surface = pygame.Surface(self.frame_size)
            blank_surface.fill((30, 30, 30))
            return blank_surface, action, finger_count, hand_present

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = Image(image_format=ImageFormat.SRGB, data=rgb)
        self.timestamp_ms += 33
        result = self.landmarker.detect_for_video(mp_image, self.timestamp_ms)

        if result and result.hand_landmarks:
            hand_present = True
            hand_landmarks = result.hand_landmarks[0]
            finger_count = self._count_fingers(hand_landmarks)
            self._draw_hand(frame, hand_landmarks)
            if finger_count == 1:
                action = "jump"
            elif finger_count == 2:
                action = "duck"

        cv2.putText(
            frame,
            f"Hand: {'YES' if hand_present else 'NO'} | Fingers: {finger_count}",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
        )
        cv2.putText(
            frame,
            "1 finger=Jump | 2 fingers=Duck | Open palm=Restart",
            (10, self.frame_size[1] - 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        surface = pygame.image.frombuffer(
            frame_rgb.tobytes(), self.frame_size, "RGB"
        ).copy()
        return surface, action, finger_count, hand_present

    def _count_fingers(self, hand_landmarks):
        """Counts extended fingers based on landmark positions."""
        tips = [8, 12, 16, 20]  # Index, middle, ring, pinky
        pips = [6, 10, 14, 18]
        finger_states = [0, 0, 0, 0]
        for i, (tip, pip) in enumerate(zip(tips, pips)):
            if hand_landmarks[tip].y < hand_landmarks[pip].y:
                finger_states[i] = 1
        return sum(finger_states)

    def release(self):
        self.cap.release()
        if hasattr(self, "landmarker"):
            self.landmarker.close()

    def _draw_hand(self, frame, landmarks):
        h, w, _ = frame.shape
        points = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
        for start, end in HAND_CONNECTIONS:
            if start < len(points) and end < len(points):
                cv2.line(frame, points[start], points[end], (0, 255, 0), 2)
        for point in points:
            cv2.circle(frame, point, 4, (255, 0, 0), -1)

    def _ensure_model_file(self) -> str:
        models_dir = Path(__file__).parent / "models"
        models_dir.mkdir(parents=True, exist_ok=True)
        model_path = models_dir / "hand_landmarker.task"
        if not model_path.exists():
            urllib.request.urlretrieve(MODEL_URL, model_path)
        return str(model_path)


@dataclass
class Obstacle:
    rect: pygame.Rect
    color: Tuple[int, int, int]
    kind: str
    speed: float
    oscillation_speed: float = 0.0
    oscillation_height: float = 0.0
    base_y: int = 0
    sprite: pygame.Surface | None = None

    def update(self, dt: float):
        self.rect.x -= int(self.speed * dt)
        if self.kind == "bird" and self.oscillation_speed > 0:
            # Simple vertical bobbing for birds to add motion variety.
            ticks = pygame.time.get_ticks()
            offset = self.oscillation_height * math.sin(ticks / self.oscillation_speed)
            self.rect.centery = self.base_y + int(offset)

    def draw(self, surface: pygame.Surface):
        if self.sprite:
            surface.blit(self.sprite, self.rect)
            return

        if self.kind == "bird":
            pygame.draw.ellipse(surface, self.color, self.rect)
            wing_mid = (self.rect.centerx, self.rect.centery)
            pygame.draw.line(
                surface,
                (250, 250, 250),
                (wing_mid[0] - 12, wing_mid[1]),
                (wing_mid[0], wing_mid[1] - 8),
                3,
            )
            pygame.draw.line(
                surface,
                (250, 250, 250),
                (wing_mid[0] + 12, wing_mid[1]),
                (wing_mid[0], wing_mid[1] - 8),
                3,
            )
        else:
            pygame.draw.rect(surface, self.color, self.rect, border_radius=6)


class Dinosaur:
    """Minimalistic emoji-based Dino sprite with jump and duck states."""

    def __init__(self, ground_y: int, run_sprite: pygame.Surface, duck_sprite: pygame.Surface):
        self.ground_y = ground_y
        self.width = 60
        self.height = 70
        self.duck_height = 40
        self.rect = pygame.Rect(80, ground_y - self.height, self.width, self.height)
        self.velocity_y = 0.0
        self.gravity = 2100.0
        self.jump_force = 950.0
        self.state = "run"
        self.animation_timer = 0.0
        self.animation_index = 0
        self.run_sprite = run_sprite
        self.duck_sprite = duck_sprite

    def on_ground(self):
        return self.rect.bottom >= self.ground_y - 1

    def update(self, action: str, dt: float):
        if action == "jump" and self.on_ground():
            self.velocity_y = -self.jump_force
            self.state = "jump"
        elif action == "duck" and self.on_ground():
            self.state = "duck"
        elif self.on_ground():
            self.state = "run"

        self.velocity_y += self.gravity * dt
        self.rect.y += int(self.velocity_y * dt)
        if self.rect.bottom >= self.ground_y:
            self.rect.bottom = self.ground_y
            self.velocity_y = 0

        target_height = self.duck_height if self.state == "duck" and self.on_ground() else self.height
        if self.rect.height != target_height:
            bottom = self.rect.bottom
            self.rect.height = target_height
            self.rect.bottom = bottom

        self.animation_timer += dt
        if self.animation_timer >= 0.15:
            self.animation_timer = 0
            self.animation_index = (self.animation_index + 1) % 2

    def draw(self, surface: pygame.Surface):
        sprite = self.duck_sprite if self.state == "duck" and self.on_ground() else self.run_sprite
        sprite_rect = sprite.get_rect()
        sprite_rect.midbottom = self.rect.midbottom
        surface.blit(sprite, sprite_rect)


class DinoGame:
    def __init__(self, width: int = 960, height: int = 540):
        pygame.init()
        pygame.display.set_caption("Gesture Dino")
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((self.width, self.height))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("Consolas", 28)
        self.large_font = pygame.font.SysFont("Consolas", 48)
        self.colors = {
            "bg": (245, 245, 245),
            "ground": (60, 60, 60),
            "score": (30, 30, 30),
            "obstacle": (34, 177, 76),
            "bird": (120, 120, 120),
        }
        self.ground_y = int(self.height * 0.8)
        dino_color = (26, 188, 102)
        run_sprite = render_ascii_sprite(DINO_ASCII_RUN, dino_color, pixel_size=6)
        duck_sprite = render_ascii_sprite(DINO_ASCII_DUCK, dino_color, pixel_size=6)
        # Flip horizontally so the dino faces left (opposite the original direction).
        self.dino_run_sprite = pygame.transform.flip(run_sprite, True, False)
        self.dino_duck_sprite = pygame.transform.flip(duck_sprite, True, False)
        self.dino = Dinosaur(self.ground_y, self.dino_run_sprite, self.dino_duck_sprite)
        self.bird_emoji = "🐦"
        self.obstacles: List[Obstacle] = []
        self.spawn_timer = 0.0
        self.spawn_delay = 2.3
        self.base_speed = 360.0
        self.speed_multiplier = 1.0
        self.score = 0.0
        self.game_over = False
        self.restart_hint_shown = False
        self.cactus_unlock_score = 600.0
        self.bird_unlock_score = 1600.0

    def reset(self):
        self.obstacles.clear()
        self.dino = Dinosaur(self.ground_y, self.dino_run_sprite, self.dino_duck_sprite)
        self.spawn_timer = 0.0
        self.spawn_delay = 2.3
        self.base_speed = 360.0
        self.speed_multiplier = 1.0
        self.score = 0.0
        self.game_over = False
        self.restart_hint_shown = False
        self.cactus_unlock_score = 600.0
        self.bird_unlock_score = 1600.0

    def update(self, action: str, dt: float):
        self.score += dt * 90
        self.speed_multiplier = 1.0 + (self.score // 200) * 0.05
        self.dino.update(action, dt)
        self._update_obstacles(dt)
        self._check_collisions()

    def _max_active_obstacles(self) -> int:
        if self.score < self.bird_unlock_score:
            return 1
        return 2

    def _update_obstacles(self, dt: float):
        self.spawn_timer += dt
        current_delay = max(0.8, self.spawn_delay - self.score / 600)
        can_spawn = len(self.obstacles) < self._max_active_obstacles()
        if self.spawn_timer >= current_delay and can_spawn:
            if self._spawn_obstacle():
                self.spawn_timer = 0
        for obstacle in list(self.obstacles):
            obstacle.update(dt)
            if obstacle.rect.right < 0:
                self.obstacles.remove(obstacle)

    def _spawn_obstacle(self):
        available_types = []
        if self.score >= self.cactus_unlock_score:
            available_types.extend(["cactus", "cactus", "cactus"])  # weight toward cactus
        if self.score >= self.bird_unlock_score:
            available_types.append("bird")

        if not available_types:
            return False

        obstacle_type = random.choice(available_types)
        speed = self.base_speed * self.speed_multiplier

        if obstacle_type == "cactus":
            sprite = render_emoji_surface("🌵", random.randint(90, 120))
            rect = sprite.get_rect()
            rect.left = self.width + 200
            rect.bottom = self.ground_y
            self.obstacles.append(
                Obstacle(
                    rect=rect.copy(),
                    color=self.colors["obstacle"],
                    kind="cactus",
                    speed=speed,
                    sprite=sprite,
                )
            )
        else:  # bird
            flight_levels = [self.ground_y - 120, self.ground_y - 170]
            y = random.choice(flight_levels)
            sprite = render_emoji_surface(self.bird_emoji, random.randint(70, 95))
            rect = sprite.get_rect()
            rect.left = self.width + 260
            rect.centery = y
            self.obstacles.append(
                Obstacle(
                    rect=rect.copy(),
                    color=self.colors["bird"],
                    kind="bird",
                    speed=speed * 1.2,
                    oscillation_speed=random.uniform(220, 280),
                    oscillation_height=random.uniform(6, 16),
                    base_y=rect.centery,
                    sprite=sprite,
                )
            )

        return True

    def _check_collisions(self):
        for obstacle in self.obstacles:
            if self.dino.rect.colliderect(obstacle.rect):
                self.game_over = True
                break

    def draw(self, camera_surface: pygame.Surface, finger_count: int, hand_present: bool):
        self.screen.fill(self.colors["bg"])
        pygame.draw.line(
            self.screen,
            self.colors["ground"],
            (0, self.ground_y),
            (self.width, self.ground_y),
            6,
        )

        self.dino.draw(self.screen)
        for obstacle in self.obstacles:
            obstacle.draw(self.screen)

        score_text = self.font.render(f"Score 🦖 {int(self.score):05d}", True, self.colors["score"])
        self.screen.blit(score_text, (30, 20))

        if camera_surface:
            preview_width = 300
            preview_height = int(preview_width * camera_surface.get_height() / camera_surface.get_width())
            preview = pygame.transform.smoothscale(camera_surface, (preview_width, preview_height))
            pygame.draw.rect(
                self.screen,
                (0, 0, 0),
                pygame.Rect(self.width - preview_width - 30, 10, preview_width + 20, preview_height + 20),
                border_radius=12,
                width=2,
            )
            self.screen.blit(preview, (self.width - preview_width - 20, 20))

        hint_text = self.font.render(
            "☝️ = Jump | ✌️ = Duck | 🖐️ = Restart", True, (80, 80, 80)
        )
        self.screen.blit(hint_text, (30, 60))

        if self.game_over:
            overlay = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 140))
            self.screen.blit(overlay, (0, 0))
            game_over_text = self.large_font.render("🦖 GAME OVER 🦖", True, (255, 255, 255))
            restart_text = self.font.render(
                "Show 🖐️ (4+ fingers) to spawn a new 🦖", True, (255, 255, 255)
            )
            self.screen.blit(
                game_over_text,
                game_over_text.get_rect(center=(self.width // 2, self.height // 2 - 30)),
            )
            self.screen.blit(
                restart_text,
                restart_text.get_rect(center=(self.width // 2, self.height // 2 + 20)),
            )
            if hand_present and finger_count >= 4:
                flashing = self.font.render("Restarting...", True, (255, 230, 0))
                self.screen.blit(
                    flashing,
                    flashing.get_rect(center=(self.width // 2, self.height // 2 + 70)),
                )

        pygame.display.flip()


def main():
    game = DinoGame()
    try:
        detector = GestureDetector()
    except RuntimeError as exc:
        print(exc)
        pygame.quit()
        sys.exit(1)
    running = True

    try:
        while running:
            dt = game.clock.tick(60) / 1000.0
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            camera_surface, action, finger_count, hand_present = detector.process_frame()

            if not game.game_over:
                game.update(action if hand_present else "run", dt)
            else:
                if hand_present and finger_count >= 4:
                    game.reset()

            game.draw(camera_surface, finger_count, hand_present)
    finally:
        detector.release()
        pygame.quit()


if __name__ == "__main__":
    main()
