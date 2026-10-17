"""Infinite-scroll grid of the preprocessing thumbnails in a browser.

    uv run scripts/fomo300k/thumbnail_viewer.py <out_dir> --view ax --shuffle

`--pattern` filters the image paths (fnmatch), `--warnings` shows only images with QC warnings.
"""

import argparse
import base64
import fnmatch
import io
import json
import random
from pathlib import Path
from queue import Queue
from threading import Thread

from flask import Flask, jsonify, render_template_string
from PIL import Image, ImageDraw

GRID_CELL_WIDTH = 128
JPEG_QUALITY = 85
BATCH_SIZE = 16
VIEWS = ("ax", "cor", "sag")

app = Flask(__name__)
queue: Queue = Queue(maxsize=2048)

HTML = f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>FOMO300K thumbnails</title>
  <style>
    body {{ margin: 0; background: #111; }}
    #grid {{
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax({GRID_CELL_WIDTH}px, 1fr));
      gap: 4px;
      padding: 4px;
    }}
    img {{ width: 100%; aspect-ratio: 1 / 1; display: block; }}
    #sentinel {{ height: 1px; }}
  </style>
</head>
<body>
  <div id="grid"></div>
  <div id="sentinel"></div>
  <script>
    const PREFETCH = 3;
    const grid = document.getElementById('grid');
    const sentinel = document.getElementById('sentinel');
    const pending = [];
    let loading = false;
    let done = false;

    function enqueue() {{
      while (!done && pending.length < PREFETCH) {{
        pending.push(fetch('/batch').then(r => r.json()));
      }}
    }}

    const visible = () => sentinel.getBoundingClientRect().top < window.innerHeight;

    async function loadBatch() {{
      if (loading || done) return;
      loading = true;
      while (!done && visible()) {{
        enqueue();
        const images = await pending.shift();
        if (images.length === 0) {{
          done = true;
          observer.disconnect();
          break;
        }}
        for (const b64 of images) {{
          const img = document.createElement('img');
          img.src = 'data:image/jpeg;base64,' + b64;
          grid.appendChild(img);
        }}
      }}
      loading = false;
    }}

    const observer = new IntersectionObserver(entries => {{
      if (entries[0].isIntersecting) loadBatch();
    }});
    observer.observe(sentinel);
  </script>
</body>
</html>"""


@app.get("/")
def index():
    return render_template_string(HTML)


@app.get("/batch")
def batch():
    images = []
    while len(images) < BATCH_SIZE:
        thumbnail = queue.get()
        if thumbnail is None:
            queue.put(None)
            break
        buffer = io.BytesIO()
        thumbnail.save(buffer, "JPEG", quality=JPEG_QUALITY)
        images.append(base64.b64encode(buffer.getvalue()).decode())
    return jsonify(images)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--view", choices=VIEWS, default="ax")
    parser.add_argument("--pattern", help="fnmatch pattern on the info.json paths")
    parser.add_argument("--warnings", action="store_true", help="only images with QC warnings")
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--port", type=int, default=5023)
    args = parser.parse_args()

    info_paths = sorted(str(path) for path in args.out_dir.rglob("*_info.json"))
    if args.pattern:
        info_paths = fnmatch.filter(info_paths, args.pattern)
    if args.shuffle:
        random.seed(args.seed)
        random.shuffle(info_paths)
    print(f"found {len(info_paths)} images")

    def produce() -> None:
        for info_path in info_paths:
            info = json.loads(Path(info_path).read_text())
            if args.warnings and not info["warnings"]:
                continue
            stem = info_path.removesuffix("_info.json")
            thumbnail = Image.open(f"{stem}_{args.view}.jpg")
            thumbnail.load()
            relative = Path(stem).relative_to(args.out_dir)
            label = f"{relative.parts[0][:24]} {relative.name.split('_')[-1]}"
            if info["warnings"]:
                label += " " + ",".join(info["warnings"])
            ImageDraw.Draw(thumbnail).text((3, 2), label, fill=255)
            queue.put(thumbnail)
        queue.put(None)

    Thread(target=produce, daemon=True).start()
    app.run(host="127.0.0.1", port=args.port)


if __name__ == "__main__":
    main()
