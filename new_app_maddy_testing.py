# new_app_maddy.py
# Streamlit app that loads a pre-trained XGBoost model from xgb_model.pkl

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pydeck as pdk  # kept in case you want maps later
from scipy import sparse
import pickle
import requests


with st.sidebar:
    st.markdown("""
        <div style="text-align:center; margin-bottom:15px;">
            <h2 style="margin-bottom:0;">🚦 Accident Risk Explorer</h2>
            <p style="font-size:14px; margin-top:5px; color:gray;">
                A Route-Based Accident Analysis & Risk Simulation App
            </p>
        </div>
        <hr style="margin:10px 0;">
        
        <div style="text-align:center; margin-top:10px;">
            <p style="font-size:13px; color:gray;">
                Developed by:<br>
                <b>Diparna Adhikary</b><br>
                <b>Madison Nommer</b><br>
                <span style="font-size:12px;">MSU Data Science</span>
            </p>
        </div>
        <hr style="margin:10px 0;">
    """, unsafe_allow_html=True)


page = st.sidebar.radio("Navigation", ["Prediction", "Monitoring", "Map Explorer", "Route Hotspots","Route Risk Simulator"])

# ---------- Data loading ----------

@st.cache_data
def load_data():
    df = pd.read_csv("df_clean.csv", parse_dates=["Start_Time"])
    return df

df = load_data()

# ---------- Model loading ----------

@st.cache_resource
def load_model():
    with open("xgb_model.pkl", "rb") as f:
        data = pickle.load(f)
    model = data["model"]
    encoder = data["encoder"]
    numeric_features = data["numeric_features"]
    categorical_features = data["categorical_features"]
    metrics = data["metrics"]
    return model, encoder, numeric_features, categorical_features, metrics

model, encoder, num_cols, cat_cols, model_metrics = load_model()

# ---------- UI: Prediction page ----------

if page == "Prediction":
    st.header("Accident Risk Prediction")
    st.markdown(
        "Use this page to simulate the **accident risk for a single scenario**. "
        "Choose a location, time, and weather conditions, and the XGBoost model "
        "will estimate the probability of a high-severity crash."
    )
    col1, col2 = st.columns(2)

    with col1:
        state = st.selectbox("State", sorted(df["State"].dropna().unique()))

        valid_cities = df.loc[df["State"] == state, "City"].dropna().unique()
        city = st.selectbox(
            "City",
            sorted(valid_cities) if len(valid_cities) > 0 else ["No cities available"]
        )

        # Zipcode filtered by State + City
        valid_zips = df[
            (df["State"] == state) & (df["City"] == city)
        ]["Zipcode"].dropna().unique()
        zipcode = st.selectbox(
            "Zipcode",
            sorted(valid_zips) if len(valid_zips) > 0 else ["No zipcodes available"]
        )

        day = st.selectbox("Day of Week", sorted(df["DayOfWeek"].dropna().unique()))
        hour = st.slider("Hour of Day (24h)", 0, 23, 8)

        rush_hour = 1 if hour in [7, 8, 9, 16, 17, 18] else 0
        is_weekend = 1 if day in ["Saturday", "Sunday"] else 0
        is_holiday = st.selectbox("Holiday?", ["No", "Yes"]) == "Yes"

    with col2:
        weather = st.selectbox("Weather", sorted(df["Weather_Group"].dropna().unique()))
        temp = st.number_input("Temperature (°F)", value=60.0)

        # Advanced Features
        with st.expander("Advanced Features"):
            vis = st.slider("Visibility (mi)", 0.1, 10.0, 5.0)
            humidity = st.slider("Humidity (%)", 0, 100, 70)
            wind = st.slider("Wind Speed (mph)", 0, 50, 5)
            pressure = st.number_input("Pressure (inHg)", value=30.0)

        precip = st.number_input("Precipitation (in)", value=0.0)
        distance = st.number_input("Distance (mi)", value=0.5)

        vis_bin = "Low" if vis < 1 else "Medium" if vis < 5 else "High"

    # Build single-row DataFrame for prediction
    sample = pd.DataFrame([{
        'Temperature(F)': temp,
        'Visibility(mi)': vis,
        'Humidity(%)': humidity,
        'Wind_Speed(mph)': wind,
        'Pressure(in)': pressure,
        'Precipitation(in)': precip,
        'Distance(mi)': distance,
        'Weather_Group': weather,
        'visibility_bin': vis_bin,
        'rush_hour': rush_hour,
        'is_weekend': is_weekend,
        'is_holiday': int(is_holiday),
        'State': state,
        'Zipcode': zipcode
    }])

    if st.button("🚦 Predict Accident Risk"):
        # Apply same preprocessing as during training
        X_cat = encoder.transform(sample[cat_cols].fillna("MISSING"))  # sparse
        X_num = sample[num_cols].fillna(0).values
        X_num_sparse = sparse.csr_matrix(X_num)

        X_input = sparse.hstack([X_num_sparse, X_cat]).tocsr()

        prob = model.predict_proba(X_input)[:, 1][0]
        label = "High Risk" if prob > 0.6 else "Medium Risk" if prob > 0.3 else "Low Risk"

        st.success(f"**Predicted Accident Risk:** {label}")
        st.metric("Predicted Probability", f"{prob * 100:.1f}%")
        st.info(
            f"{weather} weather, "
            f"{'rush hour' if rush_hour else 'off-peak'}, "
            f"{day} at {hour}:00 in {city}, {state}."
        )

# ---------- UI: Monitoring page ----------

elif page == "Monitoring":
    st.header(" Model Monitoring Dashboard")
    st.markdown(
        "This page summarizes how the trained **XGBoost model** is performing on held-out data. "
        "Use the metrics and confusion matrix to assess accuracy and class balance, and the "
        "feature drift check to see how input distributions change over time."
    )
    col1, col2, col3 = st.columns(3)
    col1.metric("ROC-AUC", f"{model_metrics['roc_auc']:.3f}")
    col2.metric("Accuracy", f"{model_metrics['accuracy']:.3f}")
    col3.metric("Records Used", len(df))

    st.subheader("Confusion Matrix")
    cm = model_metrics["confusion"]
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)

    st.subheader("Feature Drift Check (Example)")
    feature = st.selectbox(
        "Select feature to monitor:",
        ["Weather_Group", "Hour", "rush_hour", "is_weekend"]
    )

    # Old vs new distributions based on Year
    old = df[df["Year"] < 2022][feature].value_counts(normalize=True)
    new = df[df["Year"] >= 2022][feature].value_counts(normalize=True)

    drift_df = pd.concat([old.rename("Before"), new.rename("After")], axis=1).fillna(0)
    drift_df.plot(kind="bar", figsize=(8, 4))
    plt.title(f"Feature Distribution Shift: {feature}")
    st.pyplot(plt)

    st.caption("If large shifts are seen in input features, consider retraining the model.")


elif page == "Map Explorer":
    st.header(" Historical Accident Map")
    st.markdown(
        "Explore **historical accidents on an interactive map**. Filter by state, year, "
        "and severity level to see where crashes have occurred and how severe they were."
    )
    # ---- Filters ----
    col1, col2, col3 = st.columns(3)

    with col1:
        states = sorted(df["State"].dropna().unique().tolist())
        state_options = ["All"] + states
        default_state = "MI" if "MI" in states else "All"
        default_index = state_options.index(default_state)

        state_filter = st.selectbox(
            "State",
            state_options,
            index=default_index
        )

    with col2:
        year_min = int(df["Year"].min())
        year_max = int(df["Year"].max())
        year = st.slider("Year", year_min, year_max, year_max)

    with col3:
        severity_options = sorted(df["Severity"].dropna().unique().tolist())
        severity_filter = st.multiselect(
            "Severity",
            severity_options,
            default=severity_options
        )

    # ---- Apply filters ----
    df_map = df.copy()
    df_map = df_map[df_map["Year"] == year]
    df_map = df_map[df_map["Severity"].isin(severity_filter)]

    if state_filter != "All":
        df_map = df_map[df_map["State"] == state_filter]

    # Avoid crashing if nothing matches
    if df_map.empty:
        st.warning("No accidents found for the selected filters.")
    else:
        # ---- Color by severity ----
        color_map = {
            1: [0, 200, 0, 120],    # green
            2: [255, 215, 0, 130],  # yellow
            3: [255, 140, 0, 150],  # orange
            4: [220, 20, 60, 180],  # red
        }
        df_map = df_map.copy()
        df_map["color"] = df_map["Severity"].map(color_map)
        df_map["Start_Time_str"] = df_map["Start_Time"].dt.strftime("%Y-%m-%d %H:%M")

        # ---- Set initial view (center on filtered data) ----
        center_lat = df_map["Start_Lat"].mean()
        center_lng = df_map["Start_Lng"].mean()

        view_state = pdk.ViewState(
            latitude=float(center_lat),
            longitude=float(center_lng),
            zoom=5,
            pitch=0
        )

        layer = pdk.Layer(
            "ScatterplotLayer",
            data=df_map,
            get_position="[Start_Lng, Start_Lat]",
            get_fill_color="color",
            get_radius=60,
            pickable=True,
        )

        tooltip = {
            "html": "<b>{City}, {State}</b><br/>"
                    "Severity: {Severity}<br/>"
                    "Date: {Start_Time_str}",
            "style": {"backgroundColor": "white", "color": "black"}
        }

        st.pydeck_chart(
            pdk.Deck(
                map_style="mapbox://styles/mapbox/light-v9",
                initial_view_state=view_state,
                layers=[layer],
                tooltip=tooltip,
            )
        )

        st.caption(
            "Bubble color shows severity (green → red). "
            "Use the filters above to explore different years, states, and severities."
        )

elif page == "Route Hotspots":
    st.header(" Route Accident Hotspots (Actual Road Route)")
    st.markdown(
        "Select a **start and end city** to draw the actual driving route (via OSRM) and "
        "highlight historical accident hotspots near that corridor for a chosen year."
    )
    states = sorted(df["State"].dropna().unique().tolist())
    default_state = "MI" if "MI" in states else states[0]

    col1, col2 = st.columns(2)

    # ---- Start location ----
    with col1:
        start_state = st.selectbox(
            "Start State",
            states,
            index=states.index(default_state)
        )

        start_cities = sorted(
            df[df["State"] == start_state]["City"].dropna().unique().tolist()
        )
        start_city = st.selectbox("Start City", start_cities)

    # ---- End location ----
    with col2:
        end_state = st.selectbox(
            "End State",
            states,
            index=states.index(default_state)
        )

        end_cities = sorted(
            df[df["State"] == end_state]["City"].dropna().unique().tolist()
        )
        end_city = st.selectbox("End City", end_cities)

    # Year filter (for hotspots)
    year_min = int(df["Year"].min())
    year_max = int(df["Year"].max())
    year = st.slider("Year", year_min, year_max, year_max)

    if st.button("Show Route & Hotspots"):
        df_year = df[df["Year"] == year].copy()

        # --- Get centroids for start / end cities (fallback to all years if needed) ---
        start_points = df_year[
            (df_year["State"] == start_state) & (df_year["City"] == start_city)
        ]
        if start_points.empty:
            start_points = df[
                (df["State"] == start_state) & (df["City"] == start_city)
            ]

        end_points = df_year[
            (df_year["State"] == end_state) & (df_year["City"] == end_city)
        ]
        if end_points.empty:
            end_points = df[
                (df["State"] == end_state) & (df["City"] == end_city)
            ]

        if start_points.empty or end_points.empty:
            st.error("Not enough data to locate one of the cities.")
        else:
            start_lat = float(start_points["Start_Lat"].mean())
            start_lng = float(start_points["Start_Lng"].mean())
            end_lat = float(end_points["Start_Lat"].mean())
            end_lng = float(end_points["Start_Lng"].mean())

            # --- Call OSRM for driving route (real road path) ---
            osrm_url = (
                f"https://router.project-osrm.org/route/v1/driving/"
                f"{start_lng},{start_lat};{end_lng},{end_lat}"
                f"?overview=full&geometries=geojson"
            )

            try:
                resp = requests.get(osrm_url, timeout=10)
                resp.raise_for_status()
                data = resp.json()

                if not data.get("routes"):
                    st.error("No route found by the routing service.")
                    st.stop()

                route_geom = data["routes"][0]["geometry"]["coordinates"]  # [ [lon, lat], ... ]
            except Exception as e:
                st.error(f"Error getting route from OSRM: {e}")
                st.stop()

            # --- Build data for layers ---
            # PathLayer: needs a list-of-[lon, lat]
            route_df = pd.DataFrame({"path": [route_geom]})

            # Debug: scatter points along the route so we can SEE it
            route_points_df = pd.DataFrame(
                [{"lon": lon, "lat": lat} for lon, lat in route_geom]
            )

            # --- Hotspots: accidents near the route (simple bounding box) ---
            lons = [p[0] for p in route_geom]
            lats = [p[1] for p in route_geom]
            margin = 0.3  # degrees (~30–40 km buffer around the route)

            lng_min, lng_max = min(lons) - margin, max(lons) + margin
            lat_min, lat_max = min(lats) - margin, max(lats) + margin

            df_hotspots = df_year[
                df_year["Start_Lng"].between(lng_min, lng_max)
                & df_year["Start_Lat"].between(lat_min, lat_max)
            ].copy()

            # It’s okay if this is empty — we still show the route
            if df_hotspots.empty:
                st.warning(
                    "No accidents found near this route for the selected year. "
                    "Showing route only."
                )

            # Color by severity (only if we have hotspots)
            if not df_hotspots.empty:
                color_map = {
                    1: [0, 200, 0, 160],    # green
                    2: [255, 215, 0, 180],  # yellow
                    3: [255, 140, 0, 200],  # orange
                    4: [220, 20, 60, 220],  # red
                }
                df_hotspots["color"] = df_hotspots["Severity"].map(color_map)
                df_hotspots["Start_Time_str"] = df_hotspots["Start_Time"].dt.strftime(
                    "%Y-%m-%d %H:%M"
                )

            # Center view on route midpoint
            center_lat = float((max(lats) + min(lats)) / 2)
            center_lng = float((max(lons) + min(lons)) / 2)

            view_state = pdk.ViewState(
                latitude=center_lat,
                longitude=center_lng,
                zoom=8,   # a bit closer than before
                pitch=0,
            )

            # Very visible route line
            route_layer = pdk.Layer(
                "PathLayer",
                data=route_df,
                get_path="path",
                get_width=8,
                width_min_pixels=4,
                get_color=[0, 200, 0, 255],  # green
            )

            # Red dots along the route (helps debug if PathLayer is invisible)
            route_points_layer = pdk.Layer(
                "ScatterplotLayer",
                data=route_points_df,
                get_position="[lon, lat]",
                get_radius=40,
                get_fill_color=[255, 0, 0, 200],
            )

            layers = [route_layer, route_points_layer]

            # Hotspots layer (if any)
            if not df_hotspots.empty:
                hotspots_layer = pdk.Layer(
                    "ScatterplotLayer",
                    data=df_hotspots,
                    get_position="[Start_Lng, Start_Lat]",
                    get_fill_color="color",
                    get_radius=80,
                    pickable=True,
                )
                layers.append(hotspots_layer)

                tooltip = {
                    "html": "<b>{City}, {State}</b><br/>"
                            "Severity: {Severity}<br/>"
                            "Date: {Start_Time_str}",
                    "style": {"backgroundColor": "white", "color": "black"}
                }
            else:
                tooltip = None

            st.pydeck_chart(
                pdk.Deck(
                    map_style="mapbox://styles/mapbox/light-v9",
                    initial_view_state=view_state,
                    layers=layers,
                    tooltip=tooltip,
                )
            )

            st.caption(
                f"Green line and dots = actual route between {start_city}, {start_state} "
                f"and {end_city}, {end_state} (OSRM). "
                "Colored points, when present, are accidents near that route."
            )

elif page == "Route Risk Simulator":
    st.header(" Route Risk Simulator")
    st.markdown(
        "Simulate **relative risk along a route under custom conditions**. Specify the day, "
        "time, and weather, and the model estimates risk at multiple points along the route, "
        "color-coding segments from low to high risk."
    )
    states = sorted(df["State"].dropna().unique().tolist())
    default_state = "MI" if "MI" in states else states[0]

    col1, col2 = st.columns(2)

    # ---- Start location ----
    with col1:
        start_state = st.selectbox(
            "Start State",
            states,
            index=states.index(default_state)
        )

        start_cities = sorted(
            df[df["State"] == start_state]["City"].dropna().unique().tolist()
        )
        start_city = st.selectbox("Start City", start_cities)

    # ---- End location ----
    with col2:
        end_state = st.selectbox(
            "End State",
            states,
            index=states.index(default_state)
        )

        end_cities = sorted(
            df[df["State"] == end_state]["City"].dropna().unique().tolist()
        )
        end_city = st.selectbox("End City", end_cities)

    st.subheader("Travel Scenario")

    col3, col4, col5 = st.columns(3)

    with col3:
        day = st.selectbox("Day of Week", sorted(df["DayOfWeek"].dropna().unique()))
    with col4:
        hour = st.slider("Hour of Day (24h)", 0, 23, 8)
    with col5:
        is_holiday = st.selectbox("Holiday?", ["No", "Yes"]) == "Yes"

    rush_hour = 1 if hour in [7, 8, 9, 16, 17, 18] else 0
    is_weekend = 1 if day in ["Saturday", "Sunday"] else 0

    col6, col7 = st.columns(2)
    with col6:
        weather = st.selectbox("Weather", sorted(df["Weather_Group"].dropna().unique()))
    with col7:
        temp = st.number_input("Temperature (°F)", value=60.0)

    with st.expander("Advanced Weather Inputs"):
        vis = st.slider("Visibility (mi)", 0.1, 10.0, 5.0)
        humidity = st.slider("Humidity (%)", 0, 100, 70)
        wind = st.slider("Wind Speed (mph)", 0, 50, 5)
        pressure = st.number_input("Pressure (inHg)", value=30.0)
        precip = st.number_input("Precipitation (in)", value=0.0)

    vis_bin = "Low" if vis < 1 else "Medium" if vis < 5 else "High"

    if st.button("Simulate Route Risk"):
        # --- Get city centroids from data (all years) ---
        start_points = df[(df["State"] == start_state) & (df["City"] == start_city)]
        end_points = df[(df["State"] == end_state) & (df["City"] == end_city)]

        if start_points.empty or end_points.empty:
            st.error("Not enough data to locate one of the cities.")
        else:
            start_lat = float(start_points["Start_Lat"].mean())
            start_lng = float(start_points["Start_Lng"].mean())
            end_lat = float(end_points["Start_Lat"].mean())
            end_lng = float(end_points["Start_Lng"].mean())

            # --- Call OSRM for driving route ---
            osrm_url = (
                f"https://router.project-osrm.org/route/v1/driving/"
                f"{start_lng},{start_lat};{end_lng},{end_lat}"
                f"?overview=full&geometries=geojson"
            )

            try:
                resp = requests.get(osrm_url, timeout=10)
                resp.raise_for_status()
                data = resp.json()
                if not data.get("routes"):
                    st.error("No route found by the routing service.")
                    st.stop()
                route_geom = data["routes"][0]["geometry"]["coordinates"]  # [ [lon, lat], ... ]
            except Exception as e:
                st.error(f"Error getting route from OSRM: {e}")
                st.stop()

            # --- Sample points along the route for risk estimation ---
            n_points = min(120, len(route_geom))  # cap for performance
            if n_points < 2:
                st.error("Route is too short to analyze.")
                st.stop()

            idx = np.linspace(0, len(route_geom) - 1, n_points).astype(int)
            sampled_coords = [route_geom[i] for i in idx]

            # Build scenario dataframe for model
            scenario_rows = []
            for lon, lat in sampled_coords:
                scenario_rows.append({
                    'Temperature(F)': temp,
                    'Visibility(mi)': vis,
                    'Humidity(%)': humidity,
                    'Wind_Speed(mph)': wind,
                    'Pressure(in)': pressure,
                    'Precipitation(in)': precip,
                    'Distance(mi)': 1.0,          # assume 1-mile segment
                    'Weather_Group': weather,
                    'visibility_bin': vis_bin,
                    'rush_hour': rush_hour,
                    'is_weekend': is_weekend,
                    'is_holiday': int(is_holiday),
                    'State': start_state,         # approximate by start state
                    'Zipcode': "MISSING",         # generic; encoder will ignore unknown
                    'route_lon': lon,
                    'route_lat': lat
                })

            scenario_df = pd.DataFrame(scenario_rows)

            # --- Run through model ---
            X_cat = encoder.transform(scenario_df[cat_cols].fillna("MISSING"))
            X_num = scenario_df[num_cols].fillna(0).values
            X_num_sparse = sparse.csr_matrix(X_num)
            X_input = sparse.hstack([X_num_sparse, X_cat]).tocsr()

            probs = model.predict_proba(X_input)[:, 1]
            scenario_df["risk_prob"] = probs

            scenario_df["risk_label"] = pd.cut(
                scenario_df["risk_prob"],
                bins=[0, 0.3, 0.6, 1.0],
                labels=["Low", "Medium", "High"],
                include_lowest=True
            )

            scenario_df["risk_label"] = scenario_df["risk_label"].astype(str)

            risk_color_map = {
                "Low":    [0, 200, 0, 220],
                "Medium": [255, 215, 0, 230],
                "High":   [220, 20, 60, 255],
            }
            scenario_df["color"] = scenario_df["risk_label"].map(risk_color_map)
            scenario_df["risk_pct"] = (scenario_df["risk_prob"] * 100).round(1)

            # --- Summary metrics ---
            max_risk = scenario_df["risk_prob"].max()
            high_share = (scenario_df["risk_label"] == "High").mean()

            m1, m2 = st.columns(2)
            m1.metric("Max segment risk", f"{max_risk * 100:.1f}%")
            m2.metric("Share of route marked High risk", f"{high_share * 100:.1f}%")

            # --- Build map layers ---
            # Base route line (grey)
            route_df = pd.DataFrame({"path": [route_geom]})

            lons = [p[0] for p in route_geom]
            lats = [p[1] for p in route_geom]
            center_lat = float((max(lats) + min(lats)) / 2)
            center_lng = float((max(lons) + min(lons)) / 2)

            view_state = pdk.ViewState(
                latitude=center_lat,
                longitude=center_lng,
                zoom=8,
                pitch=0,
            )

            route_layer = pdk.Layer(
                "PathLayer",
                data=route_df,
                get_path="path",
                get_width=6,
                width_min_pixels=3,
                get_color=[80, 80, 80, 200],  # neutral grey base route
            )

            risk_points_layer = pdk.Layer(
                "ScatterplotLayer",
                data=scenario_df,
                get_position="[route_lon, route_lat]",
                get_fill_color="color",
                get_radius=70,
                pickable=True,
            )

            tooltip = {
                "html": (
                    "<b>Segment risk:</b> {risk_pct}%<br/>"
                    "Risk level: {risk_label}"
                ),
                "style": {"backgroundColor": "white", "color": "black"}
            }

            st.pydeck_chart(
                pdk.Deck(
                    map_style="mapbox://styles/mapbox/light-v9",
                    initial_view_state=view_state,
                    layers=[route_layer, risk_points_layer],
                    tooltip=tooltip,
                )
            )

            st.caption(
                "Grey line = route. Colored dots show relative risk under your chosen "
                "conditions (green → yellow → red = low → medium → high). "
                "This is a scenario-based estimate using the trained model, not a live forecast."
            )
