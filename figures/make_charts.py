"""Four figures for the post, from the official GIFT-Eval per-task CSVs.
Run: uv run --with pandas --with numpy --with matplotlib python make_charts.py"""
import os, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = os.environ.get("GIFT_EVAL_RESULTS", "gift-eval/results")
C, M = "eval_metrics/mean_weighted_sum_quantile_loss", "eval_metrics/MASE[0.5]"
# reference palette (dataviz skill), light mode; validated all-pairs for 3 slots
SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
OPEN = {"permissive": ("#2a78d6", "o", "Open weights, commercial use"),
        "capped": ("#2a78d6", "^", "Open weights, commercial use only below $20M/month revenue"),
        "noncommercial": ("#eb6834", "s", "Open weights, non-commercial"),
        "none": ("#1baf7a", "D", "No public weights")}
MODELS = [  # (dir, label, openness)
    ("Timestral-209M", "Timestral-209M", "capped"),
    ("t0-gamma", "t0-gamma", "permissive"),
    ("Morpheus-2.0", "Morpheus-2.0 (Multiscale)", "none"),
    ("TimesFM-3", "TimesFM-3", "noncommercial"),
    ("Timestral-28M", "Timestral-28M", "capped"),
    ("t0-beta", "t0-beta", "permissive"),
    ("Tabby", "Tabby (prompt-tuned)", "noncommercial"),
    ("Tabby-Pretrain", "Tabby-Pretrain", "noncommercial"),
    ("Chronos-2", "Chronos-2", "permissive"),
]
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "axes.edgecolor": GRID,
                     "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                     "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF})

def load(m): return pd.read_csv(f"{R}/{m}/all_results.csv").set_index("dataset")
sn = load("seasonal_naive")
norm = {d: load(d)[[C, M]].div(sn[[C, M]]) for d, _, _ in MODELS}
gm = {d: np.exp(np.log(norm[d]).mean()) for d in norm}
rng = np.random.default_rng(0)

def sz(mk, base=70): return base * 1.6 if mk == "^" else base  # triangles read small

def legend(fig, ax, y=0.035):
    hs = [ax.scatter([], [], c=c, marker=mk, s=sz(mk, 60), label=lab) for c, mk, lab in OPEN.values()]
    fig.legend(handles=hs, loc="lower center", bbox_to_anchor=(0.5, y), ncol=2, frameon=False,
               fontsize=9.5, labelcolor=INK2, columnspacing=1.5)

def clean(ax):
    for s in ("top", "right", "left"): ax.spines[s].set_visible(False)
    ax.tick_params(length=0)

# 1. CRPS relative to Timestral-209M with task-bootstrap 95% CI
ref = "Timestral-209M"
rows = []
for d, lab, o in MODELS:
    lr = np.log(norm[d][C] / norm[ref][C]).dropna().values
    b = np.exp([rng.choice(lr, len(lr)).mean() for _ in range(5000)])
    rows.append((lab, o, np.exp(lr.mean()), *np.percentile(b, [2.5, 97.5])))
fig, ax = plt.subplots(figsize=(9, 6.2))
y = np.arange(len(rows))[::-1]
ax.axvline(1, color=INK2, lw=1, ls=(0, (3, 3)))
for yi, (lab, o, r, lo, hi) in zip(y, rows):
    col, mk, _ = OPEN[o]
    ax.plot([lo, hi], [yi, yi], color=col, lw=2, solid_capstyle="round")
    ax.scatter(r, yi, c=col, marker=mk, s=sz(mk), zorder=3, edgecolors=SURF, linewidths=2)
    ax.text(hi + 0.006 if lab != ref else r + 0.012, yi, f"{r:.3f}", va="center", fontsize=10, color=INK2)
ax.set_yticks(y, [r[0] for r in rows], color=INK)
ax.set_xlabel("GIFT-Eval CRPS relative to Timestral-209M (ratio of geometric means; >1 = worse)")
ax.xaxis.grid(True, color=GRID); ax.set_axisbelow(True); clean(ax)
ax.set_title("Top three are inside each other's 95% intervals", loc="left", color=INK, fontsize=13, pad=12)
legend(fig, ax)
fig.text(0.01, 0.01, "Bars: 95% interval from resampling the 97 tasks (5,000 draws). Source: official GIFT-Eval per-task results.",
         fontsize=8.5, color=INK2)
fig.tight_layout(rect=(0, 0.13, 1, 1)); fig.savefig("fig1_intervals.png", dpi=200); plt.close(fig)

# 2. Rank by CRPS vs rank by MASE (slope chart)
labs = {d: (lab, o) for d, lab, o in MODELS}
rc = pd.Series({d: gm[d][C] for d in gm}).rank().astype(int)
rm = pd.Series({d: gm[d][M] for d in gm}).rank().astype(int)
fig, ax = plt.subplots(figsize=(8.5, 6.6))
for d in gm:
    lab, o = labs[d]; col, mk, _ = OPEN[o]
    ax.plot([0, 1], [rc[d], rm[d]], color=col, lw=2, alpha=0.9)
    ax.scatter([0, 1], [rc[d], rm[d]], c=col, marker=mk, s=sz(mk, 60), zorder=3, edgecolors=SURF, linewidths=2)
    ax.text(-0.04, rc[d], f"{lab}  {gm[d][C]:.3f}", ha="right", va="center", fontsize=10, color=INK)
    ax.text(1.04, rm[d], f"{gm[d][M]:.3f}  {lab}", ha="left", va="center", fontsize=10, color=INK)
ax.set_xlim(-0.9, 1.9); ax.set_ylim(len(gm) + 0.5, -0.3); ax.axis("off")
ax.text(0, 0.35, "Rank by CRPS\n(probabilistic)", ha="center", va="bottom", color=INK2, fontsize=10)
ax.text(1, 0.35, "Rank by MASE\n(point forecast)", ha="center", va="bottom", color=INK2, fontsize=10)
ax.set_title("The leader changes with the metric", loc="left", color=INK, fontsize=13, pad=4)
legend(fig, ax, 0.0)
fig.tight_layout(rect=(0, 0.08, 1, 1)); fig.savefig("fig2_rank_flip.png", dpi=200); plt.close(fig)

# 3. Domain heatmap (CRPS, gmean within domain); sequential blue, darker = higher error
dom = load("t0-gamma")["domain"]
hm_models = ["Timestral-209M", "t0-gamma", "Morpheus-2.0", "t0-beta", "Tabby-Pretrain", "Chronos-2"]
tab = pd.DataFrame({labs[d][0]: np.exp(np.log(norm[d][C]).groupby(dom).mean()) for d in hm_models})
n = dom.value_counts()
tab = tab.loc[n.sort_values(ascending=False).index]
ramp = matplotlib.colors.LinearSegmentedColormap.from_list(
    "blue", ["#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#104281"])
fig, ax = plt.subplots(figsize=(9.5, 5))
im = ax.imshow(tab.values, cmap=ramp, aspect="auto")
lo_, hi_ = tab.values.min(), tab.values.max()
for i in range(tab.shape[0]):
    best = tab.iloc[i].min()
    for j in range(tab.shape[1]):
        v = tab.iat[i, j]
        dark = (v - lo_) / (hi_ - lo_) > 0.45
        ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=10,
                color=SURF if dark else INK, fontweight="bold" if v == best else "normal")
ax.set_xticks(range(tab.shape[1]), tab.columns, rotation=20, ha="right", color=INK)
ax.set_yticks(range(tab.shape[0]), [f"{d} ({n[d]} tasks)" for d in tab.index], color=INK)
for s in ax.spines.values(): s.set_visible(False)
ax.set_xticks(np.arange(-.5, tab.shape[1]), minor=True); ax.set_yticks(np.arange(-.5, tab.shape[0]), minor=True)
ax.grid(which="minor", color=SURF, linewidth=2); ax.tick_params(which="both", length=0)
ax.set_title("CRPS by domain: gaps within a domain exceed the gap overall", loc="left", color=INK, fontsize=13, pad=12)
cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02); cb.outline.set_visible(False)
cb.set_label("CRPS vs seasonal naive (lower is better)", color=INK2)
fig.text(0.01, 0.01, "Bold = lowest in row. Domains with 6 or fewer tasks are indicative only.", fontsize=8.5, color=INK2)
fig.tight_layout(rect=(0, 0.03, 1, 1)); fig.savefig("fig3_domains.png", dpi=200); plt.close(fig)

# 4. Openness matrix (text table)
cols = ["Params", "Covariates", "Weights licence", "Commercial use", "Eval code", "Training code\n+ data generator", "Paper"]
data = [
    ["Timestral", "5M–209M", "Yes", "Modified MIT", "Below $20M/mo\nrevenue", "Yes", "No", "Not yet"],
    ["t0 (alpha/beta/gamma)", "102M–256M", "Yes", "Apache-2.0", "Yes", "Yes", "No", "Yes (alpha, beta)"],
    ["Tabby", "146M", "No", "CC-BY-NC 4.0", "No", "Yes", "Yes", "Yes"],
    ["Morpheus-2.0", "81.5M / 413M", "Yes", "None public", "No", "No", "No", "Yes"],
]
fig, ax = plt.subplots(figsize=(13.5, 2.8)); ax.axis("off")
t = ax.table(cellText=[r[1:] for r in data], rowLabels=[r[0] for r in data], colLabels=cols,
             cellLoc="center", rowLoc="left", loc="center")
t.auto_set_font_size(False); t.set_fontsize(10.5); t.scale(1, 2.0)
for (r, c), cell in t.get_celld().items():
    cell.set_edgecolor(GRID); cell.set_facecolor(SURF)
    cell.get_text().set_color(INK2 if r == 0 else INK)
    if r == 0: cell.get_text().set_fontweight("bold"); cell.set_height(cell.get_height() * 1.3)
ax.set_title("What each release lets you do (as of 29 October 2026)", loc="left", color=INK, fontsize=13)
fig.tight_layout(); fig.savefig("fig4_openness.png", dpi=200, bbox_inches="tight"); plt.close(fig)
print("wrote fig1-4")
