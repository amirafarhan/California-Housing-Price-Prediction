"""Streamlit demo: California district median house value prediction.

Run with:  streamlit run app.py
"""

import json

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

import california_housing as ch

st.set_page_config(page_title="California Housing Predictor", page_icon="🌴", layout="wide")

LABELS = {
    "MedInc": "Median income ($10,000s)",
    "HouseAge": "Median house age (years)",
    "AveRooms": "Average rooms per household",
    "AveBedrms": "Average bedrooms per household",
    "Population": "District population",
    "AveOccup": "Average household size",
    "Latitude": "Latitude",
    "Longitude": "Longitude",
}
CITIES = {  # coordinates of city centres
    "San Francisco": (37.77, -122.42), "Los Angeles": (34.05, -118.24), "San Diego": (32.72, -117.16),
    "San Jose": (37.34, -121.89), "Sacramento": (38.58, -121.49), "Fresno": (36.74, -119.79),
}
money = lambda v: f"${v * 100_000:,.0f}"


@st.cache_data
def get_data():
    return ch.load_data()


@st.cache_data
def get_metrics():
    try:
        return json.loads(ch.METRICS_PATH.read_text())
    except FileNotFoundError:
        return None


@st.cache_resource(show_spinner="Loading the model...")
def get_model(name: str):
    return ch.load_or_train(get_data(), name)


try:
    df = get_data()
    metrics = get_metrics()
    best = metrics["best"] if metrics else "GradientBoosting"
    model, how = get_model(best)
except Exception as exc:
    st.error(f"Could not load the data or model: {exc}")
    st.stop()

st.title("🌴 California Housing Price Predictor")
st.caption(f"{best} model trained on {len(df):,} census districts from the 1990 California census.")

tab_pred, tab_map, tab_model = st.tabs(["💰 Predict a district", "🗺️ Price map", "📈 Model performance"])
q = df[ch.FEATURES].quantile([0.01, 0.5, 0.99])

# ---------------- Prediction ----------------
with tab_pred:
    city = st.selectbox("Place the district near", ["Custom location"] + list(CITIES), index=1)
    lat0, lon0 = CITIES.get(city, (float(q.loc[0.5, "Latitude"]), float(q.loc[0.5, "Longitude"])))
    with st.form("district"):
        c1, c2, c3, c4 = st.columns(4)
        v = {}
        v["MedInc"] = c1.number_input(LABELS["MedInc"], 0.0, 20.0, float(round(q.loc[0.5, "MedInc"], 2)), 0.1,
                                      help="e.g. 3.5 = $35,000 median household income")
        v["HouseAge"] = c2.number_input(LABELS["HouseAge"], 1, 52, int(q.loc[0.5, "HouseAge"]), 1,
                                        help="The census caps this value at 52")
        v["AveRooms"] = c3.number_input(LABELS["AveRooms"], 0.5, 50.0, float(round(q.loc[0.5, "AveRooms"], 2)), 0.1)
        v["AveBedrms"] = c4.number_input(LABELS["AveBedrms"], 0.1, 20.0, float(round(q.loc[0.5, "AveBedrms"], 2)), 0.05)
        v["Population"] = c1.number_input(LABELS["Population"], 1, 40000, int(q.loc[0.5, "Population"]), 50)
        v["AveOccup"] = c2.number_input(LABELS["AveOccup"], 0.5, 20.0, float(round(q.loc[0.5, "AveOccup"], 2)), 0.1)
        v["Latitude"] = c3.number_input(LABELS["Latitude"], 32.5, 42.0, float(lat0), 0.01, key=f"lat-{city}")
        v["Longitude"] = c4.number_input(LABELS["Longitude"], -124.4, -114.3, float(lon0), 0.01, key=f"lon-{city}")
        go = st.form_submit_button("Predict median house value", type="primary")

    if go:
        if v["AveBedrms"] > v["AveRooms"]:
            st.error("Average bedrooms cannot exceed average rooms.")
        else:
            unusual = [LABELS[k] for k in ch.FEATURES if not q.loc[0.01, k] <= v[k] <= q.loc[0.99, k]]
            if unusual:
                st.warning("Outside the typical range of the data (less reliable): " + ", ".join(unusual))
            try:
                pred = float(ch.predict(model, v)[0])
            except Exception as exc:
                st.error(f"Prediction failed: {exc}")
            else:
                mae = metrics["results"][best]["mae"] if metrics else None
                left, right = st.columns([1, 2])
                left.metric("Predicted median house value", money(pred))
                if mae:
                    left.caption(f"Typical error in cross-validation: ±{money(mae)} (MAE)")
                if pred >= 4.99:
                    left.info("The census caps values at $500,000, so the true value may be higher.")
                near = df[(df.Latitude - v["Latitude"]).abs().le(0.15) & (df.Longitude - v["Longitude"]).abs().le(0.15)]
                if len(near):
                    left.caption(f"{len(near)} real districts within ~15 km: median value {money(near[ch.TARGET].median())}")
                else:
                    left.caption("No census districts within ~15 km of this location.")

                # what-if: income sweep at this location
                grid = np.linspace(q.loc[0.01, "MedInc"], q.loc[0.99, "MedInc"], 40)
                sweep = pd.DataFrame([{**v, "MedInc": g} for g in grid])
                curve = pd.DataFrame({"MedInc": grid, "value": ch.predict(model, sweep) * 100_000})
                chart = alt.Chart(curve).mark_line().encode(
                    x=alt.X("MedInc:Q", title=LABELS["MedInc"]),
                    y=alt.Y("value:Q", title="Predicted median value ($)", axis=alt.Axis(format="$,.0f")),
                    tooltip=[alt.Tooltip("MedInc:Q", format=".1f"), alt.Tooltip("value:Q", format="$,.0f")])
                dot = alt.Chart(pd.DataFrame({"MedInc": [v["MedInc"]], "value": [pred * 100_000]})).mark_point(
                    size=160, filled=True, color="#e4572e").encode(x="MedInc:Q", y="value:Q")
                right.altair_chart((chart + dot).properties(title="What if the median income changed?", height=300),
                                   width="stretch")

# ---------------- Map ----------------
with tab_map:
    st.caption("Every census district, coloured by its actual median house value.")
    sample = df.sample(min(len(df), 8000), random_state=0).assign(value=lambda d: d[ch.TARGET] * 100_000)
    st.altair_chart(alt.Chart(sample).mark_circle(size=12, opacity=0.6).encode(
        x=alt.X("Longitude:Q", scale=alt.Scale(domain=[-124.5, -114])),
        y=alt.Y("Latitude:Q", scale=alt.Scale(domain=[32.4, 42.1])),
        color=alt.Color("value:Q", title="Median value", scale=alt.Scale(scheme="viridis"), legend=alt.Legend(format="$,.0f")),
        tooltip=[alt.Tooltip("value:Q", format="$,.0f", title="median value"), alt.Tooltip("MedInc:Q", title="median income"),
                 "HouseAge", "Population"],
    ).properties(height=560), width="stretch")
    a, b = st.columns(2)
    a.altair_chart(alt.Chart(df.assign(value=df[ch.TARGET] * 100_000)).mark_bar().encode(
        x=alt.X("value:Q", bin=alt.Bin(maxbins=50), title="Median house value ($)", axis=alt.Axis(format="$,.0f")),
        y=alt.Y("count()", title="Districts")).properties(title="Distribution (note the spike at the $500k cap)", height=260),
        width="stretch")
    corr = df.corr()[ch.TARGET].drop(ch.TARGET).rename(index=LABELS).reset_index()
    corr.columns = ["feature", "correlation"]
    b.altair_chart(alt.Chart(corr).mark_bar(color="#2a9d8f").encode(
        x=alt.X("correlation:Q", scale=alt.Scale(domain=[-1, 1]), title="Correlation with median value"),
        y=alt.Y("feature:N", sort="-x", title=None)).properties(title="Feature correlations", height=260), width="stretch")

# ---------------- Model ----------------
with tab_model:
    if metrics:
        r = metrics["results"][best]
        c1, c2, c3 = st.columns(3)
        c1.metric("Cross-validated R²", f"{r['r2']:.3f}", f"± {r['r2_std']:.3f}", delta_color="off")
        c2.metric("RMSE", money(r["rmse"]))
        c3.metric("MAE", money(r["mae"]))
        res = pd.DataFrame(metrics["results"]).T.reset_index(names="model")
        st.altair_chart(alt.Chart(res).mark_bar().encode(
            x=alt.X("r2:Q", title="R² (5-fold cross-validation)", scale=alt.Scale(domain=[0, 1])),
            y=alt.Y("model:N", sort="-x", title=None),
            color=alt.condition(alt.datum.model == best, alt.value("#2a9d8f"), alt.value("#adb5bd")),
            tooltip=["model", alt.Tooltip("r2:Q", format=".3f"), alt.Tooltip("rmse:Q", format=".3f"),
                     alt.Tooltip("mae:Q", format=".3f")]).properties(height=220), width="stretch")
        st.caption("Models listed in the project README, compared on the same 5 folds. "
                   "Recompute with `python california_housing.py`.")
    if hasattr(model, "feature_importances_"):
        imp = pd.Series(model.feature_importances_, index=ch.FEATURES).rename(index=LABELS).sort_values().reset_index()
        imp.columns = ["feature", "importance"]
        st.altair_chart(alt.Chart(imp).mark_bar(color="#2a9d8f").encode(
            x=alt.X("importance:Q", title="Importance"), y=alt.Y("feature:N", sort="-x", title=None)
        ).properties(title=f"Feature importance ({best})", height=260), width="stretch")
    st.caption(f"Model {how} at startup.")
