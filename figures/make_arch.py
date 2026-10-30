"""Architecture figures: layer stacks and attention reach, from EVIDENCE.md.
Run: uv run --with matplotlib python make_arch.py"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle

SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
TIME, VAR, BOTH = "#2a78d6", "#eb6834", "#1baf7a"  # validated reference palette slots
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "figure.facecolor": SURF,
                     "axes.facecolor": SURF, "savefig.facecolor": SURF})

# 1. Layer stacks: one block per layer, coloured by what the layer attends across.
MODELS = [  # name, sub, layer pattern, width note
    ("Timestral-209M", "causal decoder, 24 layers, d=768", ["T", "T", "T", "TV"] * 6),
    ("t0-gamma", "causal decoder, 24 layers, d=1024", ["T", "T", "V"] * 8),
    ("Tabby", "encoder, 20 layers, d=768", ["T"] * 20),
    ("Morpheus-2.0 base", "encoder, 9 layers, d=640", ["B"] * 9),
]
COL = {"T": TIME, "V": VAR, "B": BOTH}
fig, ax = plt.subplots(figsize=(10, 6.4))
ax.set_xlim(0, 4); ax.set_ylim(-3.2, 28.5); ax.axis("off")
for k, (name, sub, pat) in enumerate(MODELS):
    x0 = k + 0.18
    for i, kind in enumerate(pat):
        if kind == "TV":  # time attention then variate attention inside one layer
            ax.add_patch(Rectangle((x0, i), 0.32, 0.82, color=TIME))
            ax.add_patch(Rectangle((x0 + 0.32, i), 0.32, 0.82, color=VAR))
        else:
            ax.add_patch(Rectangle((x0, i), 0.64, 0.82, color=COL[kind]))
    ax.text(k + 0.5, -0.9, name, ha="center", va="top", fontsize=11, color=INK, fontweight="bold")
    ax.text(k + 0.5, -2.0, sub, ha="center", va="top", fontsize=9, color=INK2)
ax.text(0, 28.3, "Each block is one transformer layer, bottom to top", fontsize=10, color=INK2, va="top")
for j, (c, lab) in enumerate([(TIME, "attends along time (one series)"),
                              (VAR, "attends across series (one time step)"),
                              (BOTH, "both, mixed by a learned gate")]):
    ax.add_patch(Rectangle((0.02 + 1.35 * j, 25.6), 0.12, 0.7, color=c))
    ax.text(0.18 + 1.35 * j, 25.95, lab, fontsize=9.5, color=INK, va="center")
ax.set_title("Where each model lets series talk to each other", loc="left", color=INK, fontsize=13)
fig.tight_layout(); fig.savefig("fig_arch_layers.png", dpi=200); plt.close(fig)

# 2. Attention reach on the running example: load (target) + temperature (known future).
# One forecast token (load, first future patch) and what it can read in one layer.
fig, axs = plt.subplots(1, 4, figsize=(12, 3.6))
NCTX, NFUT = 6, 1
cases = [
    ("Timestral, t0 (time layers)", {("load", c) for c in range(NCTX)}),
    ("Timestral, t0 (cross-series layers)", {("temp", NCTX)}),
    ("Tabby (load only)", {("load", c) for c in range(NCTX + NFUT)} - {("load", NCTX)}),
    ("Morpheus (every layer)", {("load", c) for c in range(NCTX)} | {("temp", NCTX)}),
]
for ax, (title, reads) in zip(axs, cases):
    rows = ["temp", "load"] if not title.startswith("Tabby") else ["load"]
    for r, row in zip(([0, 1] if len(rows) == 2 else [1]), rows):
        for c in range(NCTX + NFUT):
            q = (row == "load" and c == NCTX)
            face = INK if q else (TIME if (row, c) in reads and row == "load" else VAR if (row, c) in reads else GRID)
            ax.add_patch(FancyBboxPatch((c + 0.08, r + 0.08), 0.84, 0.84, boxstyle="round,pad=0,rounding_size=0.12",
                                        color=face, ec=SURF))
        ax.text(-0.15, r + 0.5, row, ha="right", va="center", fontsize=9.5, color=INK)
    ax.axvline(NCTX, color=INK2, lw=1, ls=(0, (3, 3)))
    ax.set_xlim(-1.3, NCTX + NFUT + 0.1); ax.set_ylim(-0.6, 2.2); ax.axis("off")
    ax.text(NCTX / 2, -0.35, "history (6 patches)", ha="center", fontsize=8.5, color=INK2)
    ax.text(NCTX + 0.75, -0.35, "next day", ha="center", fontsize=8.5, color=INK2)
    ax.set_title(title, fontsize=10.5, color=INK, loc="left")
fig.suptitle("What the forecast token (black) can read in one layer: one week of hourly load plus a temperature forecast",
             x=0.01, ha="left", fontsize=11.5, color=INK)
fig.tight_layout(rect=(0, 0, 1, 0.93)); fig.savefig("fig_arch_reach.png", dpi=200); plt.close(fig)
print("wrote fig_arch_layers.png fig_arch_reach.png")

# 3. Interest: arXiv papers per year mentioning "time series foundation model" (title or abstract).
# Counts from the arXiv API on 2026-10-30 (scratchpad arxiv_counts.py); 2026 runs to 30 October.
years, n = ["2023", "2024", "2025", "2026\n(to 30 Oct)"], [5, 43, 151, 273]
fig, ax = plt.subplots(figsize=(7.5, 4.2))
bars = ax.bar(years, n, color=TIME, width=0.6)
for b, v in zip(bars, n):
    ax.text(b.get_x() + b.get_width() / 2, v + 5, str(v), ha="center", fontsize=10.5, color=INK)
for s in ("top", "right", "left"): ax.spines[s].set_visible(False)
ax.spines["bottom"].set_color(GRID); ax.set_yticks([]); ax.tick_params(length=0, colors=INK2)
ax.set_title('arXiv papers with "time series foundation model" in the title or abstract', loc="left",
             fontsize=11.5, color=INK)
fig.text(0.01, 0.01, "Source: arXiv API, counted by first-version date, 30 October 2026.", fontsize=8.5, color=INK2)
fig.tight_layout(rect=(0, 0.04, 1, 1)); fig.savefig("fig_interest.png", dpi=200); plt.close(fig)
