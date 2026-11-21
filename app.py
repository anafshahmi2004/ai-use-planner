# AI Usage Planner - Streamlit app.
# Run from this folder with:  streamlit run app.py
# It reads model.json (exported by the notebook), so it needs no scikit-learn.
import json
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

MODEL = json.loads((Path(__file__).parent / "model.json").read_text())
H_MIN, H_MAX = MODEL["h_min"], MODEL["h_max"]
TH = MODEL["overreliance"]["thresholds"]
BLUE, ORANGE, MUTED = "#2a78d6", "#eb6834", "#898781"

st.set_page_config(page_title="AI Usage Planner", layout="wide")


def predict(key, hours):
    """Expected value plus the lower and upper edge of the 80% range."""
    m = MODEL["models"][key]
    h = np.asarray(hours, dtype=float)
    p = m["intercept"] + sum(c * h ** (i + 1) for i, c in enumerate(m["coefs"]))
    return p, p + m["lo"], p + m["hi"]


def hours_needed(target):
    grid = np.linspace(H_MIN, H_MAX, 751)
    mid, low_edge, high_edge = predict("time_saved", grid)

    def first(arr):
        ok = arr >= target
        return grid[np.argmax(ok)] if ok.any() else np.nan

    typical, fewest, most = first(mid), first(high_edge), first(low_edge)
    if np.isnan(typical):
        return (f"{target:g} h/week is beyond what the data covers "
                f"(the model reaches about {mid.max():.1f} h/week at {H_MAX:g} h/day).")
    most_txt = f"{most:.1f}" if not np.isnan(most) else f"more than {H_MAX:g}"
    return (f"To save about {target:g} h/week, people in this data typically use AI "
            f"about {typical:.1f} h/day (80% of them between {fewest:.1f} and {most_txt} h/day).")


def classify(hours, accuracy):
    if hours >= TH["heavy_hours"] and accuracy <= TH["very_low_acc"]:
        return "High concern"
    if hours >= TH["heavy_hours"] and accuracy <= TH["low_acc"]:
        return "Watch"
    return "No flag"


st.title("AI Usage Planner")
st.caption(
    f"Built from {MODEL['n_rows']} self-reported survey responses from students and professionals. "
    "Shows associations in this sample, not proof that more AI use causes the results."
)

tab_plan, tab_check, tab_data = st.tabs(["Plan my usage", "Over-reliance check", "What the data shows"])

# ------------------------------------------------------------------ planner
with tab_plan:
    left, right = st.columns([1, 2], gap="large")
    with left:
        hours = st.slider("AI hours per day", float(H_MIN), float(H_MAX), 3.0, 0.25, key="plan_hours")
        ts, ts_lo, ts_hi = (float(x) for x in predict("time_saved", hours))
        pr, pr_lo, pr_hi = (float(x) for x in predict("productivity", hours))
        c1, c2 = st.columns(2)
        c1.metric("Time saved per week", f"{max(ts, 0):.1f} h")
        c2.metric("Productivity (1-10)", f"{min(max(pr, 1), 10):.1f}")
        st.caption(
            f"80% of people land between {max(ts_lo, 0):.1f} and {ts_hi:.1f} h saved per week, "
            f"and between {min(max(pr_lo, 1), 10):.1f} and {min(max(pr_hi, 1), 10):.1f} on productivity."
        )
        st.metric("Saved per AI hour", f"{max(ts, 0) / (hours * 7):.2f} h")
        st.divider()
        target = st.number_input("I want to save (hours per week)", 1.0, 30.0, 10.0, 0.5, key="plan_target")
        st.info(hours_needed(target))
    with right:
        metric = st.radio("Chart", ["Time saved per week", "Productivity score"], horizontal=True,
                          label_visibility="collapsed", key="plan_metric")
        key, col = (("time_saved", "time_saved") if metric == "Time saved per week"
                    else ("productivity", "productivity"))
        grid = np.linspace(H_MIN, H_MAX, 120)
        mid, lo, hi = predict(key, grid)
        line = pd.DataFrame({"hours": grid, "expected": mid, "low": np.maximum(lo, 0), "high": hi})
        pts = pd.DataFrame(MODEL["points"])[["hours", col]].dropna()
        pts.columns = ["hours", "value"]
        pts = pts[pts["hours"] <= H_MAX]   # a few respondents use AI more than the chart range
        ylab = "Time saved (hours per week)" if key == "time_saved" else "Productivity score (1-10)"
        x = alt.X("hours:Q", title="AI usage (hours per day)", scale=alt.Scale(domain=[0, H_MAX + 0.4]))
        band = alt.Chart(line).mark_area(opacity=0.18, color=BLUE).encode(
            x=x, y=alt.Y("low:Q", title=ylab), y2="high:Q")
        curve = alt.Chart(line).mark_line(color=BLUE, strokeWidth=3).encode(x=x, y="expected:Q")
        dots = alt.Chart(pts).mark_circle(size=40, opacity=0.4, color=MUTED).encode(
            x=x, y=alt.Y("value:Q", title=ylab))
        here = pd.DataFrame({"hours": [hours], "value": [ts if key == "time_saved" else pr]})
        marker = alt.Chart(here).mark_point(size=140, filled=True, color=ORANGE, stroke="white",
                                            strokeWidth=2).encode(x=x, y="value:Q")
        st.altair_chart((band + dots + curve + marker).properties(height=420), width="stretch")
        st.caption(f"Grey dots are real respondents, the blue line is the expected value, the shaded band is the "
                   f"80% range and the orange dot is your selection. Valid for {H_MIN:g} to {H_MAX:g} hours a day; "
                   "above 6 hours there are few respondents.")

# ------------------------------------------------------------------ over-reliance check
with tab_check:
    st.write(
        "A quick screening rule: heavy AI use combined with low trust in the answers can mean "
        "output is being used without enough checking."
    )
    c1, c2 = st.columns(2)
    with c1:
        h2 = st.slider("AI hours per day", 0.5, 10.0, 3.0, 0.25, key="chk_hours")
    with c2:
        acc = st.select_slider("How accurate do you find the AI's answers? (1 = rarely, 5 = almost always)",
                               options=[1, 2, 3, 4, 5], value=3, key="chk_acc")
    tier = classify(h2, acc)
    if tier == "High concern":
        st.error(f"High concern: {h2:g} h/day with accuracy rated {acc}. Heavy use of answers you rate as unreliable.")
    elif tier == "Watch":
        st.info(f"Watch: {h2:g} h/day with accuracy rated {acc}. Heavy use with middling trust in the answers. "
                "In the survey this group was no less satisfied, so take it as a reminder to spot-check.")
    else:
        st.success(f"No flag: {h2:g} h/day with accuracy rated {acc}.")
    if tier != "No flag":
        st.markdown(
            "- Check a sample of the AI's output against a trusted source before using it.\n"
            "- Try a task by hand now and then to keep the skill and judge the AI's result.\n"
            "- Note which kinds of task give wrong answers and stop using AI for those."
        )
    st.subheader("How the groups compare in the survey")
    t = pd.DataFrame(MODEL["overreliance"]["tiers"]).set_index("tier")
    show = t[["n", "ai_hours", "accuracy", "productivity", "satisfaction", "pct_recommend", "pct_not_recommend"]].copy()
    show.columns = ["Respondents", "AI hrs/day", "Accuracy (1-5)", "Productivity", "Satisfaction",
                    "Would recommend", "Would not recommend"]
    show.index.name = "Group"
    for c in ["Would recommend", "Would not recommend"]:
        show[c] = (show[c] * 100).round(0).astype(int).astype(str) + "%"
    one_dp = st.column_config.NumberColumn(format="%.1f")
    st.dataframe(show, width="stretch", column_config={
        "AI hrs/day": one_dp, "Accuracy (1-5)": one_dp, "Productivity": one_dp, "Satisfaction": one_dp})
    st.caption(MODEL["overreliance"]["note"])

# ------------------------------------------------------------------ data insights
with tab_data:
    st.subheader("Payoff per AI hour falls as usage rises")
    b = pd.DataFrame(MODEL["bands"])
    b["label"] = b["band"] + "  (n=" + b["n"].astype(str) + ")"
    base = alt.Chart(b).encode(
        x=alt.X("label:N", sort=list(b["label"]), title="Daily AI usage band",
                axis=alt.Axis(labelAngle=0)),
        y=alt.Y("efficiency:Q", title="Hours saved per week per AI hour used"),
    )
    bars = base.mark_bar(color=BLUE, size=70).encode(tooltip=["band", "n", "efficiency"])
    values = base.mark_text(dy=-8, color="#52514e").encode(text=alt.Text("efficiency:Q", format=".2f"))
    st.altair_chart((bars + values).properties(height=320), width="stretch")
    st.subheader("Key findings")
    for line_ in MODEL["findings"]:
        st.markdown(f"- {line_}")
    st.subheader("Read with care")
    for line_ in MODEL["caveats"]:
        st.markdown(f"- {line_}")
