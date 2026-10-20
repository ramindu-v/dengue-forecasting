"""
app.py — Sri Lanka Dengue Early Warning dashboard

    streamlit run app.py

Reads the files produced by train.py (outputs/), so it starts instantly and needs no training.
"""

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).parent
OUT = ROOT / "outputs"

# ----------------------------------------------------------------- design tokens
# Dark theme
INK, MUTED, LINE, BG, CARD = "#E5E9F2", "#94A3B8", "#1E293B", "#070B14", "#0F1626"
BLUE, BLUE_SOFT = "#60A5FA", "rgba(96,165,250,0.18)"
AMBER, AMBER_SOFT = "#FBBF24", "rgba(251,191,36,0.13)"
GRID = "#1A2335"
ALERT = {
    "High":     {"color": "#F87171", "bg": "rgba(248,113,113,0.14)", "text": "#FCA5A5", "icon": "🔴", "order": 0},
    "Elevated": {"color": "#FBBF24", "bg": "rgba(251,191,36,0.14)", "text": "#FCD34D", "icon": "🟠", "order": 1},
    "Normal":   {"color": "#34D399", "bg": "rgba(52,211,153,0.13)", "text": "#6EE7B7", "icon": "🟢", "order": 2},
    "Unknown":  {"color": "#64748B", "bg": "rgba(100,116,139,0.18)", "text": "#CBD5E1", "icon": "⚪", "order": 3},
}
FEATURE_NAMES = {
    "y_lag0": "Cases this week", "y_lag1": "Cases 1 wk ago", "y_lag2": "Cases 2 wks ago",
    "y_lag3": "Cases 3 wks ago", "y_lag4": "Cases 4 wks ago", "y_lag5": "Cases 5 wks ago",
    "y_lag6": "Cases 6 wks ago", "y_lag7": "Cases 7 wks ago", "y_mean4": "Avg last 4 wks",
    "y_mean8": "Avg last 8 wks", "y_mean13": "Avg last 13 wks", "y_mean26": "Avg last 26 wks",
    "y_std8": "Volatility (8 wks)", "y_trend4": "4-week trend", "y_hist_mean": "District's usual level",
    "nat_lag0": "National cases", "nat_trend4": "National trend", "woy_sin": "Time of year (sin)",
    "woy_cos": "Time of year (cos)", "y_last_year": "Same week last year",
    "y_last_year_mean3": "Same season last year", "district": "District",
    "rain_0_3": "Rain 0–3 wks ago", "rain_4_7": "Rain 4–7 wks ago", "rain_8_11": "Rain 8–11 wks ago",
    "temp_0_3": "Temp 0–3 wks ago", "temp_4_7": "Temp 4–7 wks ago",
    "hum_0_3": "Humidity 0–3 wks ago", "hum_4_7": "Humidity 4–7 wks ago",
}

st.set_page_config(page_title="Dengue Early Warning · Sri Lanka", page_icon="🦟", layout="wide")

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
html, body, .stApp, .stMarkdown, button, input, textarea, select,
[data-testid="stWidgetLabel"], [data-testid="stMetric"] {{ font-family: 'Plus Jakarta Sans', sans-serif !important; }}
.stApp {{ background: {BG}; }}
.block-container {{ padding-top: 1.2rem; padding-bottom: 3rem; max-width: 1320px; }}
header[data-testid="stHeader"] {{ background: transparent; }}
#MainMenu, footer {{ visibility: hidden; }}

/* hero */
.hero {{ background: radial-gradient(900px 380px at 90% -30%, rgba(59,130,246,.55) 0%, transparent 60%),
          radial-gradient(700px 300px at -10% 120%, rgba(16,185,129,.18) 0%, transparent 60%),
          linear-gradient(135deg, #0B1324 0%, #111C36 60%, #172554 100%);
        border: 1px solid #1E2A44; border-radius: 22px; padding: 30px 34px 26px; color: #fff; margin-bottom: 18px;
        box-shadow: 0 24px 60px -30px rgba(59,130,246,.45); }}
.hero h1 {{ font-size: 2.05rem; font-weight: 800; letter-spacing: -.02em; margin: 6px 0 4px; color:#fff; }}
.hero p {{ color: #CBD5E1; margin: 0; font-size: .98rem; max-width: 760px; }}
.eyebrow {{ font-size: .75rem; letter-spacing: .14em; text-transform: uppercase; color: #93C5FD; font-weight: 700; }}
.chips {{ display: flex; gap: 8px; flex-wrap: wrap; margin-top: 14px; }}
.chip {{ background: rgba(255,255,255,.09); border: 1px solid rgba(255,255,255,.16); color: #E2E8F0;
         padding: 5px 12px; border-radius: 999px; font-size: .8rem; font-weight: 500; }}
.kpis {{ display: grid; grid-template-columns: repeat(4, minmax(0,1fr)); gap: 12px; margin-top: 22px; }}
.kpi {{ background: rgba(255,255,255,.07); border: 1px solid rgba(255,255,255,.12); border-radius: 16px; padding: 14px 16px; }}
.kpi .label {{ color: #94A3B8; font-size: .78rem; font-weight: 600; }}
.kpi .value {{ font-size: 1.75rem; font-weight: 800; letter-spacing: -.02em; margin-top: 2px; color: #fff; }}
.kpi .sub {{ color: #A5B4FC; font-size: .78rem; }}
@media (max-width: 800px) {{ .kpis {{ grid-template-columns: repeat(2, minmax(0,1fr)); }} .hero h1 {{ font-size: 1.5rem; }} }}

/* tabs as pills (works with old and new Streamlit tab markup) */
[data-testid="stTabs"] [role="tablist"] {{ gap: 8px; background: transparent; border: none !important;
    box-shadow: none !important; flex-wrap: wrap; }}
[data-testid="stTabs"] [role="tablist"]::after, [data-testid="stTabs"] > div:first-child {{ border-bottom: none !important;
    box-shadow: none !important; }}
[data-testid="stTabs"] [role="tab"] {{ background: {CARD}; border: 1px solid {LINE}; border-radius: 999px;
    padding: 8px 18px !important; height: auto; margin: 0 !important; transition: all .15s ease; }}
[data-testid="stTabs"] [role="tab"]:hover {{ border-color: #334155; }}
[data-testid="stTabs"] [role="tab"] p {{ font-weight: 600; font-size: .92rem; color: {INK}; }}
[data-testid="stTabs"] [role="tab"][aria-selected="true"] {{ background: {BLUE}; border-color: {BLUE}; }}
[data-testid="stTabs"] [role="tab"][aria-selected="true"] p {{ color: #0B1120 !important; }}
.react-aria-SelectionIndicator, div[data-baseweb="tab-highlight"], div[data-baseweb="tab-border"] {{ display: none !important; }}
[data-testid="stTabs"] [role="tabpanel"] {{ padding-top: 14px; }}

/* cards */
div[data-testid="stVerticalBlockBorderWrapper"] {{ background: {CARD}; border-radius: 18px !important;
    border-color: {LINE} !important; box-shadow: 0 10px 30px -18px rgba(0,0,0,.8); }}
.card-title {{ font-weight: 700; color: {INK}; font-size: 1.02rem; margin-bottom: 2px; }}
.card-sub {{ color: {MUTED}; font-size: .83rem; margin-bottom: 10px; }}

.banner {{ border-radius: 14px; padding: 13px 16px; font-size: .93rem; margin: 4px 0 14px;
           border: 1px solid; display: flex; gap: 10px; align-items: flex-start; }}

.badge {{ display: inline-block; padding: 3px 10px; border-radius: 999px; font-size: .74rem; font-weight: 700; }}

.dcard {{ border: 1px solid {LINE}; border-left: 5px solid; border-radius: 14px; padding: 12px 14px;
          margin-bottom: 10px; background: #0B1220; }}
.dcard .top {{ display: flex; justify-content: space-between; align-items: center; }}
.dcard .name {{ font-weight: 700; color: {INK}; }}
.dcard .nums {{ display: flex; align-items: baseline; gap: 8px; margin-top: 6px; }}
.dcard .big {{ font-size: 1.45rem; font-weight: 800; color: {INK}; letter-spacing: -.02em; }}
.dcard .range {{ color: {MUTED}; font-size: .8rem; }}
.bar {{ height: 7px; background: #1E293B; border-radius: 999px; margin-top: 8px; position: relative; overflow: hidden; }}
.bar > span {{ position: absolute; left: 0; top: 0; bottom: 0; border-radius: 999px; }}
.bar-note {{ color: {MUTED}; font-size: .74rem; margin-top: 4px; }}
.scroll {{ max-height: 575px; overflow-y: auto; padding-right: 4px; }}

.stat {{ background: {CARD}; border: 1px solid {LINE}; border-radius: 16px; padding: 14px 16px; height: 100%; }}
.stat .label {{ color: {MUTED}; font-size: .78rem; font-weight: 600; }}
.stat .value {{ color: {INK}; font-size: 1.55rem; font-weight: 800; letter-spacing: -.02em; margin-top: 2px; }}
.stat .sub {{ color: {MUTED}; font-size: .78rem; }}
.good {{ color: #4ADE80 !important; font-weight: 700; }}
.scroll::-webkit-scrollbar {{ width: 8px; }}
.scroll::-webkit-scrollbar-thumb {{ background: #1E293B; border-radius: 8px; }}

.footer {{ color: {MUTED}; font-size: .8rem; text-align: center; margin-top: 30px; }}
.footer a {{ color: {MUTED}; }}
</style>
""", unsafe_allow_html=True)


# ----------------------------------------------------------------- data
@st.cache_data
def load():
    fc = pd.read_csv(OUT / "forecast_latest.csv", parse_dates=["origin_week", "target_week"])
    hist = pd.read_csv(OUT / "history.csv", parse_dates=["week_start"])
    bt = pd.read_csv(OUT / "backtest.csv", parse_dates=["week_start", "target_week"])
    imp = pd.read_csv(OUT / "feature_importance.csv")
    coords = pd.read_csv(ROOT / "data" / "district_coords.csv")
    metrics = json.loads((OUT / "metrics.json").read_text())
    return fc, hist, bt, imp, coords, metrics


fc, hist, bt, imp, coords, metrics = load()
latest = fc.origin_week.max()
has_weather = metrics["data"]["weather"]
districts = sorted(fc.district.unique())


# ----------------------------------------------------------------- helpers
def badge(level: str) -> str:
    a = ALERT[level]
    return f'<span class="badge" style="background:{a["bg"]};color:{a["text"]}">{level}</span>'


def style_fig(fig, height=420, legend_top=True):
    fig.update_layout(
        height=height, template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Plus Jakarta Sans, sans-serif", color=INK, size=12),
        margin=dict(l=8, r=8, t=10 if not legend_top else 40, b=8),
        hoverlabel=dict(bgcolor="#0F1626", bordercolor="#334155", font=dict(color=INK)),
        legend=dict(orientation="h", y=1.08, x=0, bgcolor="rgba(0,0,0,0)") if legend_top else None,
    )
    fig.update_xaxes(showgrid=False, linecolor=LINE, tickfont=dict(color=MUTED))
    fig.update_yaxes(gridcolor=GRID, zeroline=False, tickfont=dict(color=MUTED))
    return fig


def card_header(title, sub=None):
    st.markdown(f'<div class="card-title">{title}</div>' +
                (f'<div class="card-sub">{sub}</div>' if sub else ""), unsafe_allow_html=True)


def stat(label, value, sub="", sub_class=""):
    return (f'<div class="stat"><div class="label">{label}</div><div class="value">{value}</div>'
            f'<div class="sub {sub_class}">{sub}</div></div>')


# ----------------------------------------------------------------- hero
h4 = fc[fc.horizon == 4]
n_high, n_elev = int((h4.alert == "High").sum()), int((h4.alert == "Elevated").sum())
gain4 = metrics["improvement_vs_naive_pct"]["4"]
st.markdown(f"""
<div class="hero">
  <div class="eyebrow">🦟 Public-health early warning</div>
  <h1>Dengue Forecasts for Sri Lanka</h1>
  <p>Weekly dengue cases forecast 1–4 weeks ahead for all {metrics['data']['districts']} health districts,
     with alerts when a district is heading above its usual level for the season.</p>
  <div class="chips">
    <span class="chip">📅 Data to week of {latest:%d %b %Y}</span>
    <span class="chip">🏥 Ministry of Health surveillance, 2007–2026</span>
    <span class="chip">🌧️ Weather features: {"on" if has_weather else "off"}</span>
    <span class="chip">🤖 LightGBM · tested on 2022–2025</span>
  </div>
  <div class="kpis">
    <div class="kpi"><div class="label">High-alert districts</div><div class="value">{n_high}</div>
      <div class="sub">4 weeks ahead</div></div>
    <div class="kpi"><div class="label">Elevated districts</div><div class="value">{n_elev}</div>
      <div class="sub">4 weeks ahead</div></div>
    <div class="kpi"><div class="label">Forecast cases, week of {h4.target_week.iloc[0]:%d %b}</div>
      <div class="value">{h4.point.sum():,.0f}</div><div class="sub">all districts · latest week {h4.cases_latest.sum():,.0f}</div></div>
    <div class="kpi"><div class="label">Accuracy vs naive forecast</div><div class="value">+{gain4:.0f}%</div>
      <div class="sub">4 weeks ahead · backtest</div></div>
  </div>
</div>
""", unsafe_allow_html=True)

tab_now, tab_district, tab_perf, tab_about = st.tabs(
    ["Forecast map", "District explorer", "Model performance", "About"])

# ================================================================= TAB 1
with tab_now:
    h = st.segmented_control("Forecast horizon", [1, 2, 3, 4], default=4,
                             format_func=lambda x: f"{x} wk{'s' if x > 1 else ''} ahead",
                             label_visibility="collapsed") or 4
    now = fc[fc.horizon == h].merge(coords, on="district")
    target = now.target_week.iloc[0]
    now["order"] = now.alert.map(lambda a: ALERT[a]["order"])
    now = now.sort_values(["order", "point"], ascending=[True, False])

    high = now[now.alert == "High"].district.tolist()
    elev = now[now.alert == "Elevated"].district.tolist()
    if high:
        a = ALERT["High"]
        msg = (f"<b>{len(high)} district{'s' if len(high) > 1 else ''} on high alert</b> for the week of "
               f"{target:%d %b %Y}: {', '.join(high)}. Forecast cases are at least 1.5× the usual upper level.")
    elif elev:
        a = ALERT["Elevated"]
        msg = f"<b>{len(elev)} districts elevated</b> for the week of {target:%d %b %Y}. No district is on high alert."
    else:
        a = ALERT["Normal"]
        msg = f"<b>All districts within their usual range</b> for the week of {target:%d %b %Y}."
    st.markdown(f'<div class="banner" style="background:{a["bg"]};border-color:{a["color"]}33;color:{a["text"]}">'
                f'<span>{a["icon"]}</span><span>{msg}</span></div>', unsafe_allow_html=True)

    left, right = st.columns([1.35, 1], gap="medium")
    with left:
        with st.container(border=True):
            card_header("Forecast map", f"Bubble size = forecast cases · week of {target:%d %b %Y}")
            fig = px.scatter_map(
                now, lat="lat", lon="lon", size="point", color="alert", size_max=36,
                color_discrete_map={k: v["color"] for k, v in ALERT.items()},
                category_orders={"alert": ["High", "Elevated", "Normal", "Unknown"]},
                hover_name="district",
                hover_data={"point": ":.0f", "low": ":.0f", "high": ":.0f", "upper": ":.0f",
                            "lat": False, "lon": False, "alert": False},
                labels={"point": "Forecast", "low": "Low (10%)", "high": "High (90%)", "upper": "Usual upper"},
                zoom=6.2, center={"lat": 7.85, "lon": 80.75}, map_style="carto-darkmatter")
            fig.update_traces(marker=dict(opacity=0.85))
            style_fig(fig, height=560, legend_top=False)
            fig.update_layout(legend=dict(orientation="h", y=0.02, x=0.02, title=None,
                                          bgcolor="rgba(11,18,32,0.85)", bordercolor=LINE, borderwidth=1),
                              margin=dict(l=0, r=0, t=0, b=0))
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    with right:
        with st.container(border=True):
            card_header("Priority districts", "Sorted by alert level, then forecast size")
            cards = []
            for r in now.itertuples():
                a = ALERT[r.alert]
                ratio = r.point / r.upper if r.upper and r.upper > 0 else 0
                width = min(ratio / 2, 1) * 100          # bar full at 2× the usual upper level
                cards.append(f"""
                <div class="dcard" style="border-left-color:{a['color']}">
                  <div class="top"><span class="name">{r.district}</span>{badge(r.alert)}</div>
                  <div class="nums"><span class="big">{r.point:,.0f}</span>
                    <span class="range">cases · range {r.low:,.0f}–{r.high:,.0f}</span></div>
                  <div class="bar"><span style="width:{width:.0f}%;background:{a['color']}"></span></div>
                  <div class="bar-note">{ratio:.1f}× usual upper level ({r.upper:,.0f}) · latest week {r.cases_latest:,.0f}</div>
                </div>""")
            st.markdown('<div class="scroll">' + "".join(cards) + "</div>", unsafe_allow_html=True)

    with st.expander("How to read the alerts"):
        st.markdown(
            "- **Usual level / upper level** come from the epidemiological *endemic channel*: the median and "
            "75th percentile of the same 5-week window over the previous 5 years.\n"
            "- **Elevated:** a rise above the usual upper level is likely.\n"
            "- **High:** the central forecast is at least 1.5× the usual upper level.\n"
            "- **Range:** the model's 80% prediction interval (10th to 90th percentile).")

# ================================================================= TAB 2
with tab_district:
    c1, c2 = st.columns([1, 1])
    district = c1.selectbox("District", districts, index=districts.index("Colombo"))
    years_back = c2.select_slider("History", options=[1, 2, 3, 5, 10], value=2,
                                  format_func=lambda y: f"{y} year{'s' if y > 1 else ''}")

    fd = fc[fc.district == district].sort_values("target_week")
    last4 = fd[fd.horizon == 4].iloc[0]
    hd = hist[(hist.district == district) & (hist.week_start > latest - pd.DateOffset(years=years_back))]
    change = (last4.point / last4.cases_latest - 1) * 100 if last4.cases_latest else 0

    s1, s2, s3, s4 = st.columns(4)
    s1.markdown(stat("Latest reported week", f"{last4.cases_latest:,.0f}", f"week of {latest:%d %b %Y}"),
                unsafe_allow_html=True)
    s2.markdown(stat("Forecast in 4 weeks", f"{last4.point:,.0f}",
                     f"{change:+.0f}% · range {last4.low:,.0f}–{last4.high:,.0f}"), unsafe_allow_html=True)
    s3.markdown(stat("Usual upper level", f"{last4.upper:,.0f}", f"usual level {last4.typical:,.0f}"),
                unsafe_allow_html=True)
    s4.markdown(stat("Alert in 4 weeks", badge(last4.alert), "endemic channel"), unsafe_allow_html=True)
    st.write("")

    with st.container(border=True):
        card_header(f"{district}: reported cases, usual range and forecast",
                    "Shaded band = usual range for the season (median to 75th percentile of the previous 5 years)")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=hd.week_start, y=hd.upper, line=dict(width=0), mode="lines", showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=hd.week_start, y=hd.typical, fill="tonexty", fillcolor=AMBER_SOFT, mode="lines",
                                 line=dict(width=0), name="Usual range"))
        fig.add_trace(go.Scatter(x=hd.week_start, y=hd.upper, name="Usual upper level",
                                 line=dict(color=AMBER, dash="dot", width=1.6)))
        fig.add_trace(go.Scatter(x=hd.week_start, y=hd.cases, name="Reported cases",
                                 line=dict(color=INK, width=2.2, shape="spline", smoothing=0.4)))
        fx = pd.concat([pd.DataFrame({"target_week": [latest], "point": [last4.cases_latest],
                                      "low": [last4.cases_latest], "high": [last4.cases_latest]}),
                        fd[["target_week", "point", "low", "high"]]])
        fig.add_trace(go.Scatter(x=fx.target_week, y=fx.high, line=dict(width=0), mode="lines", showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=fx.target_week, y=fx.low, fill="tonexty", fillcolor=BLUE_SOFT, mode="lines",
                                 line=dict(width=0), name="80% forecast range"))
        fig.add_trace(go.Scatter(x=fx.target_week, y=fx.point, name="Forecast", mode="lines+markers",
                                 line=dict(color=BLUE, width=3), marker=dict(size=7)))
        fig.add_vline(x=latest, line=dict(color=LINE, width=1.5, dash="dash"))
        style_fig(fig, height=440)
        fig.update_layout(hovermode="x unified", yaxis_title="Cases per week")
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    if has_weather and "rain_mm" in hd:
        with st.container(border=True):
            card_header("Rainfall and dengue", "Rain creates breeding sites; cases usually rise a few weeks later")
            wfig = go.Figure()
            wfig.add_trace(go.Bar(x=hd.week_start, y=hd.rain_mm, name="Weekly rainfall (mm)",
                                  marker_color="#2563EB", yaxis="y2", opacity=0.7))
            wfig.add_trace(go.Scatter(x=hd.week_start, y=hd.cases, name="Reported cases",
                                      line=dict(color=INK, width=2)))
            style_fig(wfig, height=320)
            wfig.update_layout(hovermode="x unified", yaxis=dict(title="Cases"),
                               yaxis2=dict(title="Rain (mm)", overlaying="y", side="right", showgrid=False))
            st.plotly_chart(wfig, use_container_width=True, config={"displayModeBar": False})

# ================================================================= TAB 3
with tab_perf:
    st.markdown(
        f'<div class="banner" style="background:rgba(96,165,250,.10);border-color:rgba(96,165,250,.35);color:#BFDBFE">'
        f'<span>🧪</span><span><b>Honest backtest.</b> For each year {metrics["test_years"][0]}–{metrics["test_years"][-1]}, '
        f'the model was trained only on earlier data, then asked to forecast every week of that year, exactly as it '
        f'would be used in real life. It is compared with simple baselines any useful model must beat.</span></div>',
        unsafe_allow_html=True)

    mae = pd.DataFrame(metrics["mae_by_horizon"]).T
    mae.index = mae.index.astype(int)
    cols = st.columns(4)
    for i, hh in enumerate(mae.index):
        cols[i].markdown(stat(f"{hh} week{'s' if hh > 1 else ''} ahead · avg error",
                              f"{mae.loc[hh, 'point']:.1f} <span style='font-size:.9rem;color:{MUTED}'>cases</span>",
                              f"▲ {metrics['improvement_vs_naive_pct'][str(hh)]:.1f}% better than naive", "good"),
                         unsafe_allow_html=True)
    st.write("")

    c1, c2 = st.columns([1.25, 1], gap="medium")
    with c1:
        with st.container(border=True):
            card_header("Forecast error by horizon", "Mean absolute error, cases per district per week (lower is better)")
            names = {"point": "LightGBM model", "naive": "Same as this week", "mean4": "Avg of last 4 weeks",
                     "seasonal_naive": "Same week last year"}
            long = mae.reset_index(names="horizon").melt("horizon", var_name="method", value_name="mae")
            long["method"] = long.method.map(names)
            long["horizon"] = long.horizon.map(lambda x: f"{x} wk")
            bar = px.bar(long, x="horizon", y="mae", color="method", barmode="group",
                         color_discrete_sequence=[BLUE, "#64748B", "#475569", "#334155"],
                         labels={"horizon": "", "mae": "MAE", "method": ""})
            bar.update_traces(marker_line_width=0)
            style_fig(bar, height=360)
            bar.update_layout(legend=dict(title=None, orientation="h", y=-0.12, x=0), bargap=0.25,
                              margin=dict(l=8, r=8, t=10, b=8))
            st.plotly_chart(bar, use_container_width=True, config={"displayModeBar": False})
    with c2:
        with st.container(border=True):
            card_header("Outbreak alerts", "Will a district go above its usual upper level?")
            rows = []
            for hh in [1, 2, 3, 4]:
                a = metrics["alerts"][f"h{hh}"]
                rows.append({"Weeks ahead": hh, "Model F1": a["model"]["f1"], "Naive F1": a["naive"]["f1"],
                             "Model recall": a["model"]["recall"], "Naive recall": a["naive"]["recall"]})
            al = pd.DataFrame(rows)
            st.dataframe(al.style.format({c: "{:.2f}" for c in al.columns if c != "Weeks ahead"}),
                         hide_index=True, use_container_width=True)
            st.caption("**Recall** = share of real rises flagged in advance. The alert threshold was tuned on "
                       "2019–2021, never on the test years.")
            st.caption(f"80% ranges contained the real value **{metrics['interval_coverage_80']:.0%}** of the time.")

    c1, c2 = st.columns([1, 1.25], gap="medium")
    with c1:
        with st.container(border=True):
            card_header("What the model relies on", "Share of total importance (gain)")
            hsel = st.segmented_control("Horizon ", [1, 4], default=4, key="imp_h",
                                        format_func=lambda x: f"{x} wk ahead", label_visibility="collapsed") or 4
            top = imp[imp.horizon == hsel].nlargest(10, "share").iloc[::-1].copy()
            top["feature"] = top.feature.map(lambda f: FEATURE_NAMES.get(f, f))
            ifig = px.bar(top, x="share", y="feature", orientation="h", color_discrete_sequence=[BLUE],
                          labels={"share": "", "feature": ""})
            ifig.update_traces(marker_line_width=0)
            style_fig(ifig, height=370, legend_top=False)
            ifig.update_xaxes(tickformat=".0%")
            st.plotly_chart(ifig, use_container_width=True, config={"displayModeBar": False})
    with c2:
        with st.container(border=True):
            card_header("Forecast vs reality", "Forecasts made in the backtest, compared with what was reported")
            k1, k2, k3 = st.columns(3)
            d2 = k1.selectbox("District ", districts, index=districts.index("Colombo"))
            y2 = k2.selectbox("Test year", metrics["test_years"], index=1)
            h2 = k3.selectbox("Weeks ahead", [1, 2, 3, 4], index=3)
            b = bt[(bt.district == d2) & (bt.test_year == y2) & (bt.horizon == h2)].sort_values("target_week")
            bfig = go.Figure()
            bfig.add_trace(go.Scatter(x=b.target_week, y=b.high, line=dict(width=0), mode="lines", showlegend=False, hoverinfo="skip"))
            bfig.add_trace(go.Scatter(x=b.target_week, y=b.low, fill="tonexty", line=dict(width=0), mode="lines",
                                      fillcolor=BLUE_SOFT, name="80% range"))
            bfig.add_trace(go.Scatter(x=b.target_week, y=b.target_cases, name="Actual", line=dict(color=INK, width=2.2)))
            bfig.add_trace(go.Scatter(x=b.target_week, y=b.point, name="Forecast", line=dict(color=BLUE, width=2.2)))
            bfig.add_trace(go.Scatter(x=b.target_week, y=b.naive, name="Naive", line=dict(color="#64748B", dash="dot")))
            style_fig(bfig, height=300)
            bfig.update_layout(hovermode="x unified")
            st.plotly_chart(bfig, use_container_width=True, config={"displayModeBar": False})

    if "mae_without_weather" in metrics:
        w = metrics["mae_without_weather"]
        st.caption("Average error **without** weather features: " +
                   ", ".join(f"{k} wk {v:.2f}" for k, v in w.items()) + ".")

# ================================================================= TAB 4
with tab_about:
    c1, c2, c3 = st.columns(3, gap="medium")
    with c1:
        with st.container(border=True):
            card_header("📊 Data")
            st.markdown(
                f"- **Dengue cases:** weekly reported cases per district, "
                f"{metrics['data']['first_week'][:4]}–{metrics['data']['last_week'][:4]} "
                f"({metrics['data']['rows']:,} district-weeks). Weekly Epidemiological Reports, Epidemiology Unit, "
                "Ministry of Health, via the [denguedatahub](https://github.com/thiyangt/denguedatahub) package "
                "by Dr. Thiyanga S. Talagala.\n"
                "- **Weather:** " + ("daily rainfall, temperature and humidity from "
                                     "[NASA POWER](https://power.larc.nasa.gov/)." if has_weather
                                     else "not used in this version."))
    with c2:
        with st.container(border=True):
            card_header("⚙️ Method")
            st.markdown(
                "- One **LightGBM** model per horizon (1–4 weeks), trained on all districts together.\n"
                "- Predicts the **change** in log cases from the latest week, using only data available at "
                "forecast time.\n"
                "- **Quantile models** give the 80% range.\n"
                "- **Alerts** use the epidemiological *endemic channel*.")
    with c3:
        with st.container(border=True):
            card_header("⚠️ Limitations")
            st.markdown(
                "- Reported cases depend on testing and reporting delays.\n"
                "- Sudden surges can't be foreseen before they start.\n"
                "- Kalmunai is a health division reported separately from Ampara.\n"
                "- A portfolio project, **not** an official public-health tool.")

st.markdown('<div class="footer">Data: Epidemiology Unit, Ministry of Health, Sri Lanka · '
            '<a href="https://github.com/thiyangt/denguedatahub">denguedatahub</a> · NASA POWER</div>',
            unsafe_allow_html=True)
