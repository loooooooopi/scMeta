"""
Stacks Figure 3's two panels into the single Figure3_v2.png that main.tex
references:

  (a) Figure3a_v2.png       -- UMAP of the scMeta-graphloss embedding
  (b) Figure3b_genes_v2.png -- cross-cancer-consistent gene attribution

The per-cancer-type GO Biological Process bar chart
(Figure3b_go_pathways_v2.png, built by make_go_figure.py) was originally
stacked here as a third panel but moved to the Supplementary: at the width
Figure 3 can occupy within the 9-page limit, three stacked panels drove the
in-figure type down to roughly 2.5pt, well below a legible print size. The GO
comparison is a negative result (the two methods' significant term sets are
small and almost entirely non-overlapping), it is stated in full in the main
text, and its q-values are tabulated in Supplementary Table 7, so it does not
need main-text figure space -- whereas the cross-cancer gene-level consistency
in panel (b) is a positive finding that the main text argues from directly.
"""
from PIL import Image, ImageDraw, ImageFont

FIGDIR = '/home/wang4887/scMetas/revision3/manuscript/Figures'
panels = [
    f'{FIGDIR}/Figure3a_v2.png',
    f'{FIGDIR}/Figure3b_genes_v2.png',
]

imgs = [Image.open(p).convert('RGB') for p in panels]

# Match every panel to the first panel's width, preserving aspect ratio
target_w = imgs[0].width
scaled = []
for im in imgs:
    if im.width == target_w:
        scaled.append(im)
    else:
        s = target_w / im.width
        scaled.append(im.resize((target_w, int(im.height * s)), Image.LANCZOS))

pad = 40
white = (255, 255, 255)
total_h = sum(im.height for im in scaled) + pad * (len(scaled) - 1)
combined = Image.new('RGB', (target_w, total_h), white)

offsets = []
y = 0
for im in scaled:
    combined.paste(im, (0, y))
    offsets.append(y)
    y += im.height + pad

# Panel labels in the top-left corner of each panel, OUP style
draw = ImageDraw.Draw(combined)
try:
    font = ImageFont.truetype("/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf", 60)
except Exception:
    font = ImageFont.load_default()
for label, y0 in zip('abc', offsets):
    draw.text((10, y0 + 5), label, fill=(0, 0, 0), font=font)

outpath = f'{FIGDIR}/Figure3_v2.png'
combined.save(outpath, dpi=(300, 300))
print("saved:", outpath, combined.size)
