#!/usr/bin/env python3
"""Inject data/images.json into template.html and write dist/index.html."""
import json, pathlib, sys
root = pathlib.Path(__file__).resolve().parent.parent
tpl = root / "template.html"
data = root / "data" / "images.json"
out = root / "dist" / "APG-ASR.html"
if not tpl.exists(): sys.exit("ERROR: template.html not found at " + str(tpl))
if not data.exists(): sys.exit("ERROR: data/images.json not found. Run scripts/prep_images.py first.")
imgs = json.loads(data.read_text())
need = ["arch"] + [f"s{n}_input" for n in range(1, 6)]
missing = [k for k in need if k not in imgs]
if missing: sys.exit("ERROR: images.json is missing keys: " + ", ".join(missing))
js = "window.__IMG__=" + json.dumps({k: {"uri": v["uri"], "w": v["w"], "h": v["h"]} for k, v in imgs.items()}) + ";"
html = tpl.read_text(encoding="utf-8")
if "/*__IMAGES__*/" not in html: sys.exit("ERROR: marker /*__IMAGES__*/ not found in template.html")
out.parent.mkdir(exist_ok=True)
out.write_text(html.replace("/*__IMAGES__*/", js), encoding="utf-8")
out_index = root / "dist" / "index.html"
out_index.write_text(html.replace("/*__IMAGES__*/", js), encoding="utf-8")
print(f"OK  wrote {out} and index.html  ({out.stat().st_size // 1024} KB, {len(imgs)} images)")
