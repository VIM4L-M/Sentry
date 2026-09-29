#!/usr/bin/env python3
"""The review demo: every scene, in order, from one command (Phase 9).

Each scene opens the mission window with a different set of models and
prints what to point out while it runs. Close the window (``Escape``) to
move to the next scene.

Usage:
    python scripts/demo.py                  # every scene, in order
    python scripts/demo.py --scene 4        # start from scene 4
    python scripts/demo.py --list           # print the running order and exit
    python scripts/demo.py --device cuda
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RUNNER = PROJECT_ROOT / "scripts" / "run_simulation.py"


@dataclass(frozen=True)
class Scene:
    """One step of the demo: what it shows, how to launch it, what to say."""

    title: str
    flags: tuple[str, ...]
    talking_points: tuple[str, ...]
    needs: tuple[str, ...] = ()


SCENES: tuple[Scene, ...] = (
    Scene(
        "The disaster city and the command center",
        (),
        (
            "Fire spreads and buildings collapse into streets while the mission runs.",
            "A* re-plans around every change; the abandoned route stays greyed out.",
            "Press G: the occupancy grid the planner actually reasons over.",
        ),
    ),
    Scene(
        "Unit II - YOLOv8n: the vehicle plans on what the cameras see",
        ("--perception",),
        (
            "Four CCTV feeds run through a fine-tuned YOLOv8n (mAP50 0.99).",
            "Press G: this grid is now a belief built from detections, not ground truth.",
        ),
        ("models/yolo/labelfix/weights/best.pt",),
    ),
    Scene(
        "Unit IV - the denoising autoencoder in heavy smoke",
        (
            "--perception",
            "--denoiser",
            "models/autoencoder/sentry/best.pt",
            "--weights",
            "models/yolo/denoised/weights/best.pt",
        ),
        (
            "Frames are smoke-degraded, cleaned by a 3M-parameter conv autoencoder, then detected.",
            "At double smoke it lifts victim recall from 0.86 to 0.99.",
        ),
        ("models/autoencoder/sentry/best.pt", "models/yolo/denoised/weights/best.pt"),
    ),
    Scene(
        "Unit V - the DQN drives",
        ("--dqn", "models/dqn/sentry/best.zip"),
        (
            "A Stable-Baselines3 DQN replaces the hand-written driver, tick by tick.",
            "It sees only 7 egocentric numbers, so it learned to drive, not the map.",
        ),
        ("models/dqn/sentry/best.zip",),
    ),
    Scene(
        "Units I + III + all - full autonomy with mission control",
        ("--full",),
        (
            "Right strip: live DQN Q-values, LSTM behaviour, what the onboard camera sees,",
            "and the fusion MLP's final call.",
            "The command center's map lags reality by 3 s - press L to switch the lag off/on.",
            "Press 3 to switch fusion off: watch the collision counter climb.",
            "Press 3 again: fusion reads the camera and overrides the DQN (OVERRIDE flash).",
            "1 camera, 2 LSTM, 4 denoiser, 5 DQN: switch any model off live.",
        ),
        ("models/dqn/sentry/best.zip", "models/fusion/truth2/best.pt"),
    ),
    Scene(
        "Anna Nagar, Chennai: satellite ground and real street photos",
        ("--full", "--map", "configs/maps/osm_annanagar.yaml", "--hazard-seed", "1"),
        (
            "Real Anna Nagar streets and buildings (OpenStreetMap) on Esri satellite imagery.",
            "The vehicle drives street by street to each victim, then back to the hospital.",
            "Right column: a real Mapillary dashcam photo of where the vehicle is, and",
            "below it what the AI actually sees. The models never see the real photos.",
            "S toggles satellite, P the street photos, V the 3D view.",
        ),
        ("configs/maps/osm_annanagar.yaml", "models/dqn/sentry/best.zip"),
    ),
    Scene(
        "Chennai, 4 km of Anna Nagar: the SENTRY command center",
        (
            "--config",
            "configs/app_chennai.yaml",
            "--full",
            "--hazard-seed",
            "2",
            "--fusion-config",
            "configs/training/fusion_veto_traffic.yaml",
        ),
        (
            "Real Chennai (OpenStreetMap + satellite) with cars, autos, bikes, people, cattle.",
            "Front camera: a real street photo; stock YOLOv8 finds the real cars and people.",
            "Route comparison: 3 A* routes scored on traffic and fire risk; it drives the safest.",
            "RL panel: the DQN retrained among traffic - 4 road users hit vs 361 before.",
            "Controls: the DQN's live steering, throttle and brake scores.",
            "Press 6 to switch the emergency brake off: it still harms no one.",
            "S satellite, P street photo, [ slower, ] faster, Space pause.",
        ),
        ("configs/maps/osm_chennai.yaml", "models/dqn/traffic/best.zip"),
    ),
    Scene(
        "Chicago, 4 km of city: the autopilot view",
        (
            "--config",
            "configs/app_city.yaml",
            "--full",
            "--hazard-seed",
            "2",
            "--fusion-config",
            "configs/training/fusion_veto_traffic.yaml",
        ),
        (
            "4 x 4 km of Chicago's West Side from OpenStreetMap: 40,000 tiles, 6 victims.",
            "The camera follows the vehicle and the map turns with it, like a Tesla display.",
            "Blue ribbon: the A* route. Brackets: what the onboard model detects right now.",
            "The DQN drives with 9 egocentric numbers, so a city 67x bigger needs no retraining.",
            "350 cars and 450 pedestrians share the streets. Left column: four surround cameras.",
            "Red EMERGENCY BRAKE: the safety layer stopped for a car or a person.",
            "Press 6 to switch the brake off: the traffic-trained DQN still gives way.",
            "F whole map / drive view, S satellite, P street photos, [ slower, ] faster.",
        ),
        ("configs/maps/osm_chicago.yaml", "models/dqn/traffic/best.zip"),
    ),
    Scene(
        "Our own streets: the college area from OpenStreetMap, in 3D",
        ("--full", "--map", "configs/maps/osm_cit.yaml"),
        (
            "Real streets around Chennai Institute of Technology (Nandambakkam, near",
            "Kundrathur), imported from OpenStreetMap by scripts/import_osm_map.py.",
            "OSM has few building outlines here, so blocks between streets are inferred.",
            "No model was trained on this layout.",
            "The ground is real satellite imagery (press S to toggle); V switches to 3D.",
            "Press V: the same mission in an isometric 3D view. V again for top-down.",
            "The DQN drives egocentrically, so it transfers to streets it has never seen.",
        ),
        ("configs/maps/osm_cit.yaml", "models/dqn/sentry/best.zip"),
    ),
)


def main() -> int:
    """Run the scenes from ``--scene`` onward."""
    args = _parse_args()
    if args.list:
        for number, scene in enumerate(SCENES, start=1):
            print(f"{number}. {scene.title}")
        return 0
    for number, scene in enumerate(SCENES, start=1):
        if number < args.scene:
            continue
        _run(number, scene, args.device, args.speed)
    print("\nDemo complete.")
    return 0


def _run(number: int, scene: Scene, device: str, speed: float) -> None:
    """Print the scene's talking points, then block on its window."""
    print(f"\n=== Scene {number}/{len(SCENES)}: {scene.title} ===")
    missing = [path for path in scene.needs if not (PROJECT_ROOT / path).is_file()]
    if missing:
        print(f"  skipped - missing {', '.join(missing)} (train it first; see README)")
        return
    for point in scene.talking_points:
        print(f"  - {point}")
    print("  ([ slower, ] faster; Escape closes the window and moves on)")
    command = [
        sys.executable,
        str(RUNNER),
        *scene.flags,
        "--device",
        device,
        "--speed",
        str(speed),
    ]
    subprocess.run(command, cwd=PROJECT_ROOT, check=False)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the SENTRY AI review demo.")
    parser.add_argument("--scene", type=int, default=1, help="Start from this scene number.")
    parser.add_argument("--list", action="store_true", help="Print the scenes and exit.")
    parser.add_argument("--device", default="cpu", help='Model device ("cpu", "cuda", index).')
    parser.add_argument(
        "--speed",
        type=float,
        default=0.3,
        help="Simulation speed for every scene; 0.3 = three moves a second (default).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
