from pathlib import Path
from PIL import Image, ImageDraw

import config


def main() -> None:
    size = 64
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)

    d.ellipse((2, 2, size - 2, size - 2), fill=(255, 215, 0, 230), outline=(80, 60, 0, 255), width=2)

    d.rounded_rectangle((18, 32, 44, 54), radius=4, fill=(255, 255, 255, 255), outline=(60, 60, 60, 255), width=2)
    d.rounded_rectangle((28, 12, 40, 36), radius=5, fill=(255, 255, 255, 255), outline=(60, 60, 60, 255), width=2)
    for y in (38, 44, 50):
        d.line((20, y, 42, y), fill=(160, 160, 160, 255), width=1)

    config.ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    out = config.THUMBS_UP_PATH
    img.save(out)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()

