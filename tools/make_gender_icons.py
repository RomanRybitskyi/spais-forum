from PIL import Image, ImageDraw

import config


SIZE = 64


def _base() -> tuple[Image.Image, ImageDraw.ImageDraw]:
    img = Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0))
    return img, ImageDraw.Draw(img)


def make_male() -> Image.Image:
    img, d = _base()
    d.ellipse((2, 2, SIZE - 2, SIZE - 2),
              fill=(70, 140, 230, 235), outline=(20, 50, 120, 255), width=2)
    cx, cy, r = 26, 38, 12
    d.ellipse((cx - r, cy - r, cx + r, cy + r),
              outline=(255, 255, 255, 255), width=4)
    d.line((cx + r * 0.7, cy - r * 0.7, 52, 12),
           fill=(255, 255, 255, 255), width=4)
    d.polygon([(54, 10), (40, 12), (52, 24)],
              fill=(255, 255, 255, 255))
    return img


def make_female() -> Image.Image:
    img, d = _base()
    d.ellipse((2, 2, SIZE - 2, SIZE - 2),
              fill=(230, 90, 150, 235), outline=(110, 30, 70, 255), width=2)
    cx, cy, r = 32, 26, 12
    d.ellipse((cx - r, cy - r, cx + r, cy + r),
              outline=(255, 255, 255, 255), width=4)
    d.line((cx, cy + r, cx, 56), fill=(255, 255, 255, 255), width=4)
    d.line((cx - 8, 50, cx + 8, 50), fill=(255, 255, 255, 255), width=4)
    return img


def main() -> None:
    config.ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    male_path = config.ASSETS_DIR / "male.png"
    female_path = config.ASSETS_DIR / "female.png"
    make_male().save(male_path)
    make_female().save(female_path)
    print(f"Wrote {male_path}")
    print(f"Wrote {female_path}")


if __name__ == "__main__":
    main()

