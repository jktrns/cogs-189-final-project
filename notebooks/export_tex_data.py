import json
import re
from pathlib import Path

NB = Path("report.ipynb")
OUT = Path("data")
OUT.mkdir(exist_ok=True)

with open(NB) as f:
    nb = json.load(f)

cells = nb["cells"]


def get_stream(cell):
    for out in cell.get("outputs", []):
        if out.get("output_type") == "stream":
            return "".join(out.get("text", []))
    return ""


def get_markdown(cell):
    for out in cell.get("outputs", []):
        text = "".join(out.get("data", {}).get("text/markdown", []))
        if text:
            return text
    return ""


def parse_md_table(text):
    PLACEHOLDER = "\x00"
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    table_lines = [l for l in lines if l.startswith("|")]
    if len(table_lines) < 3:
        return []

    def split_row(line):
        safe = line.replace("\\|", PLACEHOLDER)
        return [p.strip().replace(PLACEHOLDER, "|") for p in safe.strip("|").split("|")]

    headers = split_row(table_lines[0])
    rows = []
    for line in table_lines[2:]:
        vals = split_row(line)
        rows.append(dict(zip(headers, vals)))
    return rows


def pct(s):
    return s.replace("%", "").strip()


def write_table(filename, caption, label, col_spec, header_row, body_rows, note=None):
    lines = [
        r"\begin{table}[H]",
        r"  \centering",
        f"  \\caption{{{caption}}}",
    ]
    if note:
        lines[-1] = f"  \\caption{{{caption}\n    {note}}}"
    lines += [
        f"  \\label{{{label}}}",
        f"  \\begin{{tabular}}{{{col_spec}}}",
        r"    \toprule",
        f"    {header_row} \\\\",
        r"    \midrule",
    ]
    for row in body_rows:
        lines.append(f"    {row} \\\\")
    lines += [
        r"    \bottomrule",
        r"  \end{tabular}",
        r"\end{table}",
    ]
    (OUT / filename).write_text("\n".join(lines) + "\n")
    print(f"Wrote {filename}")


md7 = get_markdown(cells[7])
rows7 = parse_md_table(md7)
ratings_rows = []
for r in rows7:
    ratings_rows.append(
        f"{r.get('Dimension','')} & {r.get('Mean','')} & {r.get('Std','')} "
        f"& {r.get('Min','')} & {r.get('Max','')} & {r.get('Median','')}"
    )
write_table(
    "table_ratings.tex",
    "Descriptive statistics of self-report ratings across all subjects and trials ($n = 1{,}280$).",
    "tab:ratings",
    "lrrrrr",
    "Dimension & Mean & Std & Min & Max & Median",
    ratings_rows,
)

md10 = get_markdown(cells[10])
rows10 = parse_md_table(md10)
like_count = dislike_count = like_pct = dislike_pct = ""
for r in rows10:
    cls = r.get("Class", "")
    if "Like" in cls and "Dis" not in cls:
        like_count = r.get("Count", "")
        like_pct = r.get("Fraction", "")
    elif "Dislike" in cls:
        dislike_count = r.get("Count", "")
        dislike_pct = r.get("Fraction", "")

md23 = get_markdown(cells[23])
rows23 = parse_md_table(md23)
s01_margin_n = s01_margin_removed = ""
for r in rows23:
    if r.get("Subject", "").strip() == "s01":
        s01_margin_n = r.get("Margin split (n)", "").strip()
        removed_str = r.get("Trials removed", "")
        m = re.search(r"\((\d+)%\)", removed_str)
        s01_margin_removed = m.group(1) if m else ""
        break

stream25 = get_stream(cells[25])
pattern = re.compile(
    r"(s\d{2})\.\.\. base=([\d.]+)% base\+margin=([\d.]+)% vote=([\d.]+)%"
)
subjects = {}
for m in pattern.finditer(stream25):
    subjects[m.group(1)] = {
        "base": float(m.group(2)),
        "margin": float(m.group(3)),
        "vote": float(m.group(4)),
    }


def cell_color(v):
    if v >= 60.0:
        return "acchi"
    elif v >= 50.0:
        return "accmid"
    else:
        return "acclo"


def fmt(sid, col):
    v = subjects[sid][col]
    c = cell_color(v)
    return f"\\cellcolor{{{c}}}{v:.1f}"


subj_ids = [f"s{i:02d}" for i in range(1, 33)]
half = len(subj_ids) // 2
persubject_rows = []
for i in range(half):
    left = subj_ids[i]
    right = subj_ids[i + half]
    persubject_rows.append(
        f"{left} & {fmt(left,'base')} & {fmt(left,'margin')} & {fmt(left,'vote')}"
        f" & {right} & {fmt(right,'base')} & {fmt(right,'margin')} & {fmt(right,'vote')}"
    )

legend = (
    r"\tightlegendbox{acchi}{green} $\geq$60\%;\enspace"
    r"\tightlegendbox{accmid}{yellow} 50--59\%;\enspace"
    r"\tightlegendbox{acclo}{red} $<$50\%."
)
write_table(
    "table_persubject.tex",
    f"Per-subject classification accuracy (\\%) across three strategies. {legend}",
    "tab:persubject",
    "cccc|cccc",
    r"Subj. & Baseline & Margin & Ensemble & Subj. & Baseline & Margin & Ensemble",
    persubject_rows,
)

md27 = get_markdown(cells[27])
rows27 = parse_md_table(md27)
strategy_map = {
    "Baseline (164-dim, median)": "Baseline (median split)",
    "Baseline + Margin binariz.": "Baseline with margin",
    "Majority Vote Ensemble": "Majority-vote ensemble",
}
agg_vals = {}
aggregate_rows = []
for r in rows27:
    strat_raw = r.get("Strategy", "").strip()
    strat = strategy_map.get(strat_raw, strat_raw)
    mean = pct(r.get("Mean accuracy", ""))
    std = pct(r.get("Std", ""))
    mn = pct(r.get("Min", ""))
    mx = pct(r.get("Max", ""))
    aggregate_rows.append(f"{strat} & {mean} & {std} & {mn} -- {mx}")
    agg_vals[strat_raw] = {"mean": mean, "std": std, "min": mn, "max": mx}
aggregate_rows.append("Chance & 50.0 & --- & ---")
write_table(
    "table_aggregate.tex",
    "Classification accuracy across 32 subjects (mean, standard deviation, and range).",
    "tab:results",
    "lccc",
    r"Strategy & Mean (\%) & Std (\%) & Range (\%)",
    aggregate_rows,
)

md31 = get_markdown(cells[31])
rows31 = parse_md_table(md31)
shap_rows = []
for r in rows31[:10]:
    rank = r.get("Rank", "").strip()
    feat = r.get("Feature", "").strip().strip("`")
    if feat.startswith("FAA_"):
        pair = feat[4:].replace("-", "--")
        feat_tex = f"FAA, {pair}"
    elif feat.startswith("DE_"):
        parts = feat[3:].split("_")
        feat_tex = "DE, " + ", ".join(parts)
    else:
        feat_tex = feat
    val = r.get("Mean |SHAP|", "").strip()
    shap_rows.append(f"{rank} & {feat_tex} & {val}")
write_table(
    "table_shap.tex",
    "Top 10 features by mean absolute SHAP value, averaged across all 32 subjects.",
    "tab:shap",
    "clr",
    r"Rank & Feature & Mean $|\mathrm{SHAP}|$",
    shap_rows,
)

md40 = get_markdown(cells[40])
rows40 = parse_md_table(md40)
temporal = {}
for r in rows40:
    key = r.get("Metric", "").strip()
    val = r.get("Value", "").strip().rstrip("d").strip()
    temporal[key] = val

first_time = temporal.get("First decodable time (>55%)", "")
peak_val_time = temporal.get("Peak accuracy", "")
final_val_time = temporal.get("Final accuracy", "")


def parse_acc_at_time(s):
    m = re.match(r"([\d.]+)%\s+at\s+([\d.]+)s", s)
    if m:
        return m.group(1), m.group(2)
    return s, ""


peak_acc, peak_time = parse_acc_at_time(peak_val_time)
final_acc, final_time = parse_acc_at_time(final_val_time)

temporal_rows = [
    r"First decodable time ($>$55\%) & " + first_time,
    f"Peak accuracy & {peak_acc}\\% at {peak_time}\\,s",
    f"Final accuracy & {final_acc}\\% at {final_time}\\,s",
]
write_table(
    "table_temporal.tex",
    "Time-resolved decoding summary.",
    "tab:temporal",
    "lr",
    "Metric & Value",
    temporal_rows,
)

md42 = get_markdown(cells[42])
rows42 = parse_md_table(md42)
condition_map = {
    "All Trials": "All trials",
    "High Familiarity": "High familiarity only",
    "Low Familiarity": "Low familiarity only",
}
familiarity_rows = []
fam_vals = {}
for r in rows42:
    cond_raw = r.get("Condition", "").strip()
    cond = condition_map.get(cond_raw, cond_raw)
    acc = pct(r.get("Accuracy", ""))
    std = pct(r.get("Std", ""))
    n = r.get("N subjects", "").strip()
    familiarity_rows.append(f"{cond} & {acc} & {std} & {n}")
    fam_vals[cond_raw] = {"acc": acc, "std": std, "n": n}
write_table(
    "table_familiarity.tex",
    "Familiarity disentanglement results.",
    "tab:familiarity",
    "lccc",
    r"Condition & Mean accuracy (\%) & Std (\%) & $n$ subjects",
    familiarity_rows,
)

base_mean = agg_vals.get("Baseline (164-dim, median)", {}).get("mean", "")
base_std  = agg_vals.get("Baseline (164-dim, median)", {}).get("std", "")
base_min  = agg_vals.get("Baseline (164-dim, median)", {}).get("min", "")
base_max  = agg_vals.get("Baseline (164-dim, median)", {}).get("max", "")
marg_mean = agg_vals.get("Baseline + Margin binariz.", {}).get("mean", "")
marg_std  = agg_vals.get("Baseline + Margin binariz.", {}).get("std", "")
marg_min  = agg_vals.get("Baseline + Margin binariz.", {}).get("min", "")
marg_max  = agg_vals.get("Baseline + Margin binariz.", {}).get("max", "")
vote_mean = agg_vals.get("Majority Vote Ensemble", {}).get("mean", "")
vote_std  = agg_vals.get("Majority Vote Ensemble", {}).get("std", "")
vote_min  = agg_vals.get("Majority Vote Ensemble", {}).get("min", "")
vote_max  = agg_vals.get("Majority Vote Ensemble", {}).get("max", "")

all_acc = fam_vals.get("All Trials", {}).get("acc", "")
hi_acc  = fam_vals.get("High Familiarity", {}).get("acc", "")
lo_acc  = fam_vals.get("Low Familiarity", {}).get("acc", "")
hi_n    = fam_vals.get("High Familiarity", {}).get("n", "")
lo_n    = fam_vals.get("Low Familiarity", {}).get("n", "")

macros = f"""\
\\newcommand{{\\BaseMean}}{{{base_mean}}}
\\newcommand{{\\BaseStd}}{{{base_std}}}
\\newcommand{{\\BaseMin}}{{{base_min}}}
\\newcommand{{\\BaseMax}}{{{base_max}}}
\\newcommand{{\\MargMean}}{{{marg_mean}}}
\\newcommand{{\\MargStd}}{{{marg_std}}}
\\newcommand{{\\MargMin}}{{{marg_min}}}
\\newcommand{{\\MargMax}}{{{marg_max}}}
\\newcommand{{\\VoteMean}}{{{vote_mean}}}
\\newcommand{{\\VoteStd}}{{{vote_std}}}
\\newcommand{{\\VoteMin}}{{{vote_min}}}
\\newcommand{{\\VoteMax}}{{{vote_max}}}
\\newcommand{{\\TemporalFirstTime}}{{{first_time}}}
\\newcommand{{\\TemporalPeakAcc}}{{{peak_acc}}}
\\newcommand{{\\TemporalPeakTime}}{{{peak_time}}}
\\newcommand{{\\TemporalFinalAcc}}{{{final_acc}}}
\\newcommand{{\\TemporalFinalTime}}{{{final_time}}}
\\newcommand{{\\FamAllAcc}}{{{all_acc}}}
\\newcommand{{\\FamHighAcc}}{{{hi_acc}}}
\\newcommand{{\\FamLowAcc}}{{{lo_acc}}}
\\newcommand{{\\FamHighN}}{{{hi_n}}}
\\newcommand{{\\FamLowN}}{{{lo_n}}}
\\newcommand{{\\LikeCount}}{{{like_count}}}
\\newcommand{{\\DislikeCount}}{{{dislike_count}}}
\\newcommand{{\\LikePct}}{{{pct(like_pct)}}}
\\newcommand{{\\DislikePct}}{{{pct(dislike_pct)}}}
\\newcommand{{\\SoneMarginN}}{{{s01_margin_n}}}
\\newcommand{{\\SoneMarginRemoved}}{{{s01_margin_removed}}}
"""

(OUT / "macros.tex").write_text(macros)
print("Wrote macros.tex")
print("\nDone.")
