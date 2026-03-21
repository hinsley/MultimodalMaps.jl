from pathlib import Path

from PIL import Image, ImageDraw


ROOT = Path("/Users/carterhinsley/Documents/Dev/MultimodalMaps.jl/kneading/experiment/attempt-011")
INPUT_NAME = "grid500_seq7_prefixes_contours.png"
OUTPUT_NAMES = (
    "grid500_seq7_prefixes_contours_hookfill.png",
    "grid500_seq7_prefixes_prefix07_contours_hookfill.png",
)

# This polygon is traced directly in image space from the visible white hook so
# the overlay follows the rendered blank region instead of a larger wedge.
HOOK_POLYGON_PIXELS = [
    (600, 1580),
    (566, 1510),
    (550, 1430),
    (544, 1345),
    (548, 1255),
    (566, 1165),
    (593, 1075),
    (630, 987),
    (673, 911),
    (719, 849),
    (769, 797),
    (812, 758),
    (835, 744),
    (831, 786),
    (808, 865),
    (781, 950),
    (753, 1040),
    (725, 1135),
    (696, 1232),
    (668, 1328),
    (640, 1420),
    (616, 1505),
]

FILL_RGBA = (244, 214, 75, 210)
OUTLINE_RGBA = (181, 138, 21, 255)
OUTLINE_WIDTH = 5
def render_highlight() -> None:
    input_path = ROOT / INPUT_NAME
    img = Image.open(input_path).convert("RGBA")
    draw = ImageDraw.Draw(img, "RGBA")
    draw.polygon(HOOK_POLYGON_PIXELS, fill=FILL_RGBA, outline=OUTLINE_RGBA, width=OUTLINE_WIDTH)

    for output_name in OUTPUT_NAMES:
        img.save(ROOT / output_name)


if __name__ == "__main__":
    render_highlight()
