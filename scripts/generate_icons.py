"""
generate_icons.py — Create tray icon PNGs for Fly on the Wall.

Run once after cloning:
    python scripts/generate_icons.py

Requires Pillow:
    pip install Pillow

Outputs:
    assets/icon-idle.png       (grey microphone — app is idle)
    assets/icon-recording.png  (red microphone — recording in progress)
"""

from pathlib import Path

try:
    from PIL import Image, ImageDraw
except ImportError:
    raise SystemExit("Pillow is required: pip install Pillow")

ASSETS_DIR = Path(__file__).parent.parent / "assets"
ASSETS_DIR.mkdir(exist_ok=True)

SIZE = 32   # 32×32; Electron resizes to 16×16 for the tray


def draw_mic(color_body: str, color_stand: str, bg: tuple) -> Image.Image:
    """Draw a simple microphone icon."""
    img = Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)

    # Mic body (rounded rectangle)
    mx, my = SIZE // 2, SIZE // 2
    bw, bh = 10, 14
    d.rounded_rectangle(
        [mx - bw // 2, 2, mx + bw // 2, 2 + bh],
        radius=5,
        fill=color_body,
    )

    # Stand arc
    stand_top = 2 + bh - 3
    stand_bot = stand_top + 8
    d.arc([mx - 8, stand_top, mx + 8, stand_bot + 4], start=0, end=180, fill=color_stand, width=2)

    # Vertical stem
    d.line([mx, stand_top + 6, mx, SIZE - 4], fill=color_stand, width=2)

    # Base
    d.line([mx - 5, SIZE - 4, mx + 5, SIZE - 4], fill=color_stand, width=2)

    return img


def main():
    # Idle: grey mic on transparent background
    idle = draw_mic(color_body="#888888", color_stand="#888888", bg=(0, 0, 0, 0))
    idle_path = ASSETS_DIR / "icon-idle.png"
    idle.save(idle_path)
    print(f"Created {idle_path}")

    # Recording: red mic — visually distinct to signal active capture
    recording = draw_mic(color_body="#E53935", color_stand="#C62828", bg=(0, 0, 0, 0))
    rec_path = ASSETS_DIR / "icon-recording.png"
    recording.save(rec_path)
    print(f"Created {rec_path}")

    print("Done. Run `npm start` to launch the app.")


if __name__ == "__main__":
    main()
