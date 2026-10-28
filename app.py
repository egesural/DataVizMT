import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx

# 1. Page Configuration
st.set_page_config(page_title="Dzud Climate Forcing Dashboard", layout="wide", initial_sidebar_state="expanded")

# ==========================================
# Variable Name Mapping
# ==========================================
variable_mapping = {
    "WINTER_PRECIP_ERA5": "Winter Precipitation (ERA5)",
    "WINTER_PRECIP_NCEP": "Winter Precipitation (NCEP)",
    "WINTER_PRECIP_MERRA2": "Winter Precipitation (MERRA-2)",
    "WINTER_PRECIP_OBS_NOV_MAR": "Winter Precip (Observed)",
    "WINTER_PRECIP_WRF": "Winter Precipitation (WRF)",
    "ColdSpell_ERA5": "Cold Spell (ERA5)",
    "ColdSpell_MERRA2": "Cold Spell (MERRA-2)",
    "ColdSpell_NCEP": "Cold Spell (NCEP)",
    "ColdSpell_OBS": "Cold Spell (Observed)",
    "ColdSpell_WRF": "Cold Spell (WRF)",
    "FTC_ERA5": "Freeze-Thaw Cycles (ERA5)",
    "FTC_MERRA2": "Freeze-Thaw Cycles (MERRA-2)",
    "FTC_NCEP": "Freeze-Thaw Cycles (NCEP)",
    "FTC_OBS": "Freeze-Thaw (Observed)",
    "FTC_WRF": "Freeze-Thaw Cycles (WRF)",
    "WCI_ERA5": "Wind Chill Index (ERA5)",
    "WCI_MERRA2": "Wind Chill Index (MERRA-2)",
    "WCI_NCEP": "Wind Chill Index (NCEP)",
    "WCI_OBS": "Wind Chill (Observed)",
    "WCI_WRF": "Wind Chill Index (WRF)",
    "SPEI_OBS": "SPEI (Observed)",
    "SPEI_ERA": "SPEI (ERA5)",
    "SPEI_MERRA": "SPEI (MERRA-2)",
    "SPEI_NCEP": "SPEI (NCEP)",
    "SPEI_WRF": "SPEI (WRF)",
    "MaxSnow_Era5": "Max Snow (ERA5)",
    "MaxSnow_Merra2": "Max Snow (MERRA-2)",
    "MaxSnow_Ncep": "Max Snow (NCEP)",
    "MaxSnow_Obs": "Max Snow (Observed)",
    "XCOLD_ERA5": "Extreme Cold (ERA5)",
    "XCOLD_MERRA2": "Extreme Cold (MERRA-2)",
    "XCOLD_NCEP": "Extreme Cold (NCEP)",
    "XCOLD_OBS": "Extreme Cold (Observed)",
    "XCOLD_WRF": "Extreme Cold (WRF)",
    "XWindDay_Era5": "Extreme Wind Days (ERA5)",
    "XWindDay_Merra2": "Extreme Wind Days (MERRA-2)",
    "XWindDay_Ncep": "Extreme Wind Days (NCEP)",
    "XWindDay_Wrf": "Extreme Wind Days (WRF)",
    "XWindDay_Obs": "Extreme Wind (Observed)",
    "Dzud_Score_ERA5": "Dzud Score (ERA5)",
    "Dzud_Score_MERRA2": "Dzud Score (MERRA-2)",
    "Dzud_Score_NCEP": "Dzud Score (NCEP)",
    "Dzud_Score_WRF": "Dzud Score (WRF)",
    "Dzud_Score_OBS": "Dzud Score (Observed)"
}

def get_long_name(col_name):
    if col_name in variable_mapping:
        return variable_mapping[col_name]
    clean_name = col_name.replace('_', ' ').title()
    for model in ["Era5", "Era", "Ncep", "Merra2", "Merra", "Wrf", "Obs"]:
        if model in clean_name:
            clean_name = clean_name.replace(model, f"({model.upper()})")
    return clean_name

def get_color_scale(var_name):
    var_lower = var_name.lower()
    if 'precip' in var_lower or 'snow' in var_lower or 'spei' in var_lower:
        return px.colors.sequential.Blues
    elif 'cold' in var_lower or 'wci' in var_lower or 'ftc' in var_lower:
        return px.colors.sequential.Purp
    elif 'dzud' in var_lower or 'wind' in var_lower:
        return px.colors.sequential.Inferno
    else:
        return px.colors.sequential.Viridis

# 2. Data Loading & Cleaning
@st.cache_data
def load_data():
    df = pd.read_csv("DZUD_INDEX_VARIABLES_ONLY.csv")
    time_cols = [col for col in df.columns if col.lower() in ['time', 'date', 'datetime', 'year']]
    time_col = time_cols[0] if time_cols else None
    
    if time_col and 'year' not in time_col.lower():
        df[time_col] = pd.to_datetime(df[time_col])
    elif time_col and 'year' in time_col.lower():
        df[time_col] = pd.to_datetime(df[time_col], format='%Y')
    
    if time_col:
        df = df.sort_values(by=time_col)
        
    cat_cols = df.select_dtypes(include=['object', 'string', 'category']).columns.tolist()
    if not cat_cols:
        cat_cols = [col for col in df.columns if 'id' in col.lower() or 'station' in col.lower() or 'name' in col.lower()]
    station_col = cat_cols[0] if cat_cols else None

    elev_cols = [col for col in df.columns if 'elev' in col.lower()]
    elev_col = elev_cols[0] if elev_cols else None

    return df, time_col, station_col, elev_col

df, time_col, station_col, elev_col = load_data()

numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
spatial_cols = [col for col in df.columns if col.lower() in ['lat', 'latitude', 'lon', 'longitude']]

exclude_keywords = ['lat', 'lon', 'latitude', 'longitude', 'index', 'synop', 'id', 'station', 'time', 'date', 'year', 'month', 'day', 'elev', 'elevation']
atmospheric_vars = [
    col for col in numeric_cols 
    if not any(keyword in col.lower() for keyword in exclude_keywords)
]

# ==========================================
# Sidebar: Global Filters
# ==========================================
with st.sidebar:
    st.header("⚙️ Global Filters")
    st.markdown("Adjust these filters to dynamically update all visual modalities simultaneously.")
    
    min_year = int(df[time_col].dt.year.min())
    max_year = int(df[time_col].dt.year.max())
    selected_years = st.slider("Select Year Range:", min_value=min_year, max_value=max_year, value=(min_year, max_year))
    
    filtered_df = df[(df[time_col].dt.year >= selected_years[0]) & (df[time_col].dt.year <= selected_years[1])].copy()
    
    all_stations = sorted(df[station_col].dropna().unique().tolist())
    selected_stations = st.multiselect("Select Specific Stations (Optional):", options=all_stations, default=[])
    
    if selected_stations:
        filtered_df = filtered_df[filtered_df[station_col].isin(selected_stations)]

    st.markdown("---")
    st.markdown("**Data Quality Profile:**")
    total_cells = filtered_df.size 
    missing_cells = filtered_df.isnull().sum().sum()
    if total_cells > 0:
        completeness = (1 - (missing_cells/total_cells)) * 100
        st.progress(completeness / 100, text=f"Data Completeness: {completeness:.1f}%")

# ==========================================
# Main Dashboard & Context
# ==========================================
st.title("Selenge River Basin: Dzud Dynamics & Climate Forcing")

with st.expander("📚 What is a Dzud & About This Research", expanded=True):
    st.markdown("""
    ### What is a Dzud?
    A **Dzud** (Zud) is a severe Mongolian winter disaster characterized by extreme cold, heavy snowfall, and impenetrable ice crusts that prevent livestock from foraging. This leads to mass animal mortality, devastating the livelihoods of nomadic herders and causing severe socioeconomic impacts across the region.

    ### About This Research
    Evaluating climate forcing uncertainties in complex terrains like the **Selenge River Basin** is critical due to the scarcity of ground meteorological observations. This project investigates the reliability and biases of various reanalysis datasets (such as **ERA5, MERRA-2, NCEP**) and regional climate models (**WRF**) in capturing extreme winter weather hazards. By exploring spatial distributions, temporal trends, and multivariate correlations, this research aims to improve the modeling, prediction, and hazard assessment of Dzud events in data-scarce regions.
    """)

st.markdown("---")

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total Stations Active", filtered_df[station_col].nunique())
with col2:
    st.metric("Data Points (N)", len(filtered_df))
with col3:
    if 'Dzud_Score_OBS' in filtered_df.columns:
        highest_dzud = filtered_df['Dzud_Score_OBS'].max()
        st.metric("Highest OBS Dzud Risk", f"{highest_dzud:.2f}")
    else:
        st.metric("Years Analyzed", selected_years[1] - selected_years[0] + 1)
with col4:
    st.metric("Elevation Range", f"{int(filtered_df[elev_col].min())}m - {int(filtered_df[elev_col].max())}m" if elev_col else "N/A")

st.markdown("---")

# 4. Multimodal Tabs
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
    "📍 Spatiotemporal Map", 
    "🏔️ 3D Terrain Analysis",
    "📈 Temporal Trends", 
    "📊 Station Profiling", 
    "🕸️ Network Topology",
    "🌡️ Correlation Matrix",
    "⚖️ Reanalysis vs OBS",
    "📉 Time Series Decomposition" # NEW TAB
])

# ==========================================
# Modality 1: Spatiotemporal Animated Maps
# ==========================================
with tab1:
    st.header("Animated Spatial Distribution")
    st.markdown("Press the **Play** button at the bottom left of the map to animate the evolution of hazards over time.")
    
    if len(spatial_cols) >= 2:
        lat_col = next(col for col in spatial_cols if 'lat' in col.lower())
        lon_col = next(col for col in spatial_cols if 'lon' in col.lower())
        
        map_var = st.selectbox("Select variable to map:", atmospheric_vars, format_func=get_long_name, key="map_var")
        
        filtered_df['Year_Frame'] = filtered_df[time_col].dt.year.astype(str)
        map_df = filtered_df.sort_values('Year_Frame')
        
        min_val = map_df[map_var].quantile(0.05)
        max_val = map_df[map_var].quantile(0.95)
        
        try:
            fig_map = px.scatter_map(
                map_df, lat=lat_col, lon=lon_col, color=map_var,
                animation_frame="Year_Frame", animation_group=station_col,
                hover_name=station_col, hover_data={lat_col: False, lon_col: False, 'Year_Frame': False, map_var: True}, 
                color_continuous_scale=get_color_scale(map_var), range_color=[min_val, max_val], 
                size_max=15, zoom=4.8, center=dict(lat=49.0, lon=104.0),
                map_style="carto-darkmatter"
            )
        except AttributeError:
            fig_map = px.scatter_mapbox(
                map_df, lat=lat_col, lon=lon_col, color=map_var,
                animation_frame="Year_Frame", animation_group=station_col,
                hover_name=station_col, hover_data={lat_col: False, lon_col: False, 'Year_Frame': False, map_var: True},
                color_continuous_scale=get_color_scale(map_var), range_color=[min_val, max_val],
                size_max=15, zoom=4.8, center=dict(lat=49.0, lon=104.0),
                mapbox_style="carto-darkmatter"
            )
            
        fig_map.update_traces(marker=dict(size=14, opacity=0.9))
        fig_map.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
        st.plotly_chart(fig_map, use_container_width=True)

# ==========================================
# Modality 2: 3D Terrain Analysis
# ==========================================
with tab2:
    st.header("3D Topographical Impact on Climate Forcing")
    st.markdown("Rotate this 3D scatter plot to explore how **Elevation (Z-Axis)** intersects with Geography (X/Y) to impact atmospheric variables.")
    
    if elev_col:
        var_3d = st.selectbox("Select variable for 3D color mapping:", atmospheric_vars, format_func=get_long_name, key="var_3d")
        
        fig_3d = px.scatter_3d(
            filtered_df, x=lon_col, y=lat_col, z=elev_col,
            color=var_3d, hover_name=station_col,
            color_continuous_scale=get_color_scale(var_3d),
            title=f"3D Representation of {get_long_name(var_3d)} Across Terrain"
        )
        fig_3d.update_traces(marker=dict(size=6, opacity=0.8))
        fig_3d.update_layout(
            margin=dict(l=0, r=0, b=0, t=40),
            scene=dict(
                xaxis_title='Longitude',
                yaxis_title='Latitude',
                zaxis_title='Elevation (m)'
            )
        )
        st.plotly_chart(fig_3d, use_container_width=True)

# ==========================================
# Modality 3: Temporal Trends
# ==========================================
with tab3:
    st.header("Temporal Evolution of Winter Hazards")
    if time_col:
        ts_var = st.selectbox("Select variable for time-series:", atmospheric_vars, format_func=get_long_name, key="ts_var")
        df_ts = filtered_df.groupby(time_col)[ts_var].mean().reset_index()
        
        fig_ts = px.line(
            df_ts, x=time_col, y=ts_var,
            title=f"Basin-wide Average Trend: {get_long_name(ts_var)}",
            color_discrete_sequence=['#0072B2'], markers=True
        )
        fig_ts.update_layout(xaxis=dict(rangeslider=dict(visible=True)))
        st.plotly_chart(fig_ts, use_container_width=True)

# ==========================================
# Modality 4: Radar & Violin Profiling
# ==========================================
with tab4:
    st.header("Station Hazard Profiling")
    col_radar, col_box = st.columns([1, 1])
    
    with col_radar:
        st.subheader("Multivariate Radar Chart")
        default_radar_vars = [v for v in atmospheric_vars if 'OBS' in v.upper()][:5]
        if len(default_radar_vars) < 3: default_radar_vars = atmospheric_vars[:5]
        
        radar_vars = st.multiselect("Select 3+ Variables for Hazard Profile:", atmospheric_vars, default=default_radar_vars, format_func=get_long_name)
        radar_stations = st.multiselect("Select Stations for Radar:", all_stations, default=all_stations[:1] if all_stations else [])
        
        if len(radar_vars) >= 3 and radar_stations:
            radar_df = filtered_df.groupby(station_col)[radar_vars].mean().reset_index()
            for var in radar_vars:
                v_min, v_max = radar_df[var].min(), radar_df[var].max()
                radar_df[var] = (radar_df[var] - v_min) / (v_max - v_min + 1e-9)
            
            fig_radar = go.Figure()
            for station in radar_stations:
                station_data = radar_df[radar_df[station_col] == station]
                if not station_data.empty:
                    vals = [station_data[var].values[0] for var in radar_vars]
                    vals += [vals[0]]
                    theta = [get_long_name(v) for v in radar_vars] + [get_long_name(radar_vars[0])]
                    
                    fig_radar.add_trace(go.Scatterpolar(
                        r=vals, theta=theta, fill='toself', name=station
                    ))
            fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])), showlegend=True, margin=dict(t=60, b=60, l=140, r=140))
            st.plotly_chart(fig_radar, use_container_width=True)
            
    with col_box:
        st.subheader("Statistical Density (Violin Plot)")
        comp_var = st.selectbox("Select variable to view extremes:", atmospheric_vars, format_func=get_long_name, key="comp_var")
        box_stations = st.multiselect("Compare Stations:", all_stations, default=all_stations[:4] if len(all_stations) >= 4 else all_stations)
        
        if box_stations:
            box_df = filtered_df[filtered_df[station_col].isin(box_stations)]
            fig_violin = px.violin(
                box_df, x=station_col, y=comp_var, color=station_col,
                box=True, points="all",
                color_discrete_sequence=px.colors.qualitative.Safe
            )
            fig_violin.update_layout(showlegend=False, xaxis_title="Station Name", yaxis_title=get_long_name(comp_var))
            st.plotly_chart(fig_violin, use_container_width=True)

# ==========================================
# Modality 5: Network Topology
# ==========================================
with tab5:
    st.header("Network Topology")
    st.markdown("Visualizing Pearson correlation to identify redundant features in the atmospheric modeling space.")
    threshold = st.slider("Correlation Threshold (Absolute Pearson):", 0.1, 1.0, 0.65, 0.05)
    
    corr_matrix = filtered_df[atmospheric_vars].corr()
    
    G = nx.Graph()
    for col in atmospheric_vars: G.add_node(get_long_name(col))
    for i in range(len(atmospheric_vars)):
        for j in range(i + 1, len(atmospheric_vars)):
            w = corr_matrix.iloc[i, j]
            if abs(w) >= threshold:
                color = '#0072B2' if w > 0 else '#D55E00'
                G.add_edge(get_long_name(atmospheric_vars[i]), get_long_name(atmospheric_vars[j]), weight=w, color=color)
                
    pos = nx.spring_layout(G, seed=42)
    fig_net = go.Figure()
    
    for edge in G.edges(data=True):
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        fig_net.add_trace(go.Scatter(
            x=[x0, x1, None], y=[y0, y1, None],
            line=dict(width=abs(edge[2]['weight']) * 5, color=edge[2]['color']),
            mode='lines', hoverinfo='none'
        ))

    fig_net.add_trace(go.Scatter(
        x=[pos[n][0] for n in G.nodes()], y=[pos[n][1] for n in G.nodes()],
        mode='markers+text', text=list(G.nodes()), textposition="top center",
        marker=dict(size=14, color='#E69F00', line_width=1), hoverinfo='text'
    ))
    
    fig_net.update_layout(
        showlegend=False,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        margin=dict(t=0, b=0, l=0, r=0)
    )
    st.plotly_chart(fig_net, use_container_width=True)

# ==========================================
# Modality 6: Correlation Heatmap
# ==========================================
with tab6:
    st.header("Correlation Matrix (Heatmap)")
    st.markdown("A full-scale view of variable interactions. High multicollinearity indicates variables that carry redundant meteorological signals.")
    
    mask_upper = st.checkbox("Mask Upper Triangle (Remove Duplicates)", value=True)
    corr_matrix_renamed = corr_matrix.rename(index=get_long_name, columns=get_long_name)
    
    if mask_upper:
        mask = np.triu(np.ones_like(corr_matrix_renamed, dtype=bool))
        corr_matrix_renamed = corr_matrix_renamed.mask(mask)
    
    fig_heat = px.imshow(
        corr_matrix_renamed,
        text_auto=".2f", aspect="auto",
        color_continuous_scale="RdBu_r", zmin=-1, zmax=1
    )
    fig_heat.update_layout(width=1200, height=900, margin=dict(t=40, b=40, l=40, r=40))
    st.plotly_chart(fig_heat, use_container_width=True)

# ==========================================
# Modality 7: Model Evaluation & Error Metrics
# ==========================================
with tab7:
    st.header("Reanalysis Model Evaluation & Bias Profiling")
    st.markdown("Evaluate the accuracy of ERA5, MERRA-2, NCEP, and WRF reanalysis models by mathematically comparing them against ground-truth station observations (OBS).")
    
    eval_categories = {
        "Winter Precipitation": ["WINTER_PRECIP_OBS_NOV_MAR", "WINTER_PRECIP_ERA5", "WINTER_PRECIP_MERRA2", "WINTER_PRECIP_NCEP", "WINTER_PRECIP_WRF"],
        "Extreme Cold (XCOLD)": ["XCOLD_OBS", "XCOLD_ERA5", "XCOLD_MERRA2", "XCOLD_NCEP", "XCOLD_WRF"],
        "Cold Spells": ["ColdSpell_OBS", "ColdSpell_ERA5", "ColdSpell_MERRA2", "ColdSpell_NCEP", "ColdSpell_WRF"],
        "Freeze-Thaw Cycles": ["FTC_OBS", "FTC_ERA5", "FTC_MERRA2", "FTC_NCEP", "FTC_WRF"],
        "Dzud Risk Score": ["Dzud_Score_OBS", "Dzud_Score_ERA5", "Dzud_Score_MERRA2", "Dzud_Score_NCEP", "Dzud_Score_WRF"]
    }
    
    eval_choice = st.selectbox("Select Hazard Variable to Evaluate:", list(eval_categories.keys()))
    eval_cols = eval_categories[eval_choice]
    valid_eval_cols = [col for col in eval_cols if col in filtered_df.columns]
    
    if len(valid_eval_cols) >= 2:
        
        # 1. TIME SERIES
        st.subheader("Time-Series Comparison")
        st.markdown("Basin-wide averages over time.")
        
        eval_time_df = filtered_df.groupby(time_col)[valid_eval_cols].mean().reset_index()
        eval_melted = eval_time_df.melt(id_vars=time_col, value_vars=valid_eval_cols, var_name="Dataset", value_name="Value")
        eval_melted["Dataset"] = eval_melted["Dataset"].apply(get_long_name)
        
        fig_eval_ts = px.line(
            eval_melted, x=time_col, y="Value", color="Dataset",
            color_discrete_sequence=px.colors.qualitative.G10, markers=True
        )
        fig_eval_ts.update_layout(legend_title="Source", margin=dict(l=0, r=0, t=30, b=0), height=500)
        st.plotly_chart(fig_eval_ts, use_container_width=True)
        
        st.markdown("---")
        
        # 2. BIAS DISTRIBUTION
        st.subheader("Model Bias Distribution (Error)")
        st.markdown("Calculated as: **Model Value - OBS Value** (Zero = Perfect Match). Wide boxes indicate high uncertainty in the reanalysis model.")
        
        obs_col = valid_eval_cols[0] 
        bias_df = pd.DataFrame()
        bias_df[station_col] = filtered_df[station_col]
        
        bias_cols = []
        for model_col in valid_eval_cols[1:]:
            bias_df[model_col] = filtered_df[model_col] - filtered_df[obs_col]
            bias_cols.append(model_col)
            
        bias_melted = bias_df.melt(id_vars=station_col, value_vars=bias_cols, var_name="Model", value_name="Error (Bias)")
        bias_melted["Model"] = bias_melted["Model"].apply(get_long_name)
        
        fig_bias = px.box(
            bias_melted, x="Model", y="Error (Bias)", color="Model",
            color_discrete_sequence=px.colors.qualitative.Pastel
        )
        fig_bias.add_hline(y=0, line_dash="dash", line_color="white", annotation_text="Zero Bias (Perfect Match)")
        fig_bias.update_layout(showlegend=False, xaxis_title="Reanalysis Dataset", margin=dict(l=0, r=0, t=30, b=0), height=600)
        st.plotly_chart(fig_bias, use_container_width=True)

        st.markdown("---")
        
        # 3. QUANTITATIVE ERROR METRICS
        st.subheader("Quantitative Error Metrics")
        st.markdown("""
        * **MAE (Mean Absolute Error):** Average magnitude of errors (Lower is better ↓).
        * **RMSE (Root Mean Square Error):** Penalizes larger errors heavily (Lower is better ↓).
        * **ACC (Anomaly Correlation):** Pearson correlation reflecting the linear relationship (Higher is better ↑).
        * **MSSS (Mean Squared Skill Score):** Compares model skill relative to observation climatology (Max 1.0, Higher is better ↑).
        """)

        metrics_data = []
        # Drop NaNs across the selected valid columns to ensure fair, aligned mathematical comparison
        eval_df_clean = filtered_df[[obs_col] + valid_eval_cols[1:]].dropna()
        
        if len(eval_df_clean) > 1:
            obs_array = eval_df_clean[obs_col].values
            obs_variance = np.mean((obs_array - np.mean(obs_array))**2) # MSE of climatology

            for model_col in valid_eval_cols[1:]:
                mod_array = eval_df_clean[model_col].values

                # MAE & RMSE
                mae = np.mean(np.abs(mod_array - obs_array))
                mse = np.mean((mod_array - obs_array)**2)
                rmse = np.sqrt(mse)

                # ACC (Pearson Correlation)
                if np.std(mod_array) == 0 or np.std(obs_array) == 0:
                    acc = np.nan
                else:
                    acc = np.corrcoef(mod_array, obs_array)[0, 1]

                # MSSS (Mean Squared Skill Score)
                if obs_variance == 0:
                    msss = np.nan
                else:
                    msss = 1 - (mse / obs_variance)

                metrics_data.append({
                    "Reanalysis Model": get_long_name(model_col),
                    "MAE ↓": mae,
                    "RMSE ↓": rmse,
                    "ACC ↑": acc,
                    "MSSS ↑": msss
                })

            # Format and highlight best metrics
            metrics_df = pd.DataFrame(metrics_data).set_index("Reanalysis Model")
            styled_metrics = metrics_df.style.format("{:.3f}") \
                .highlight_min(subset=["MAE ↓", "RMSE ↓"], color='#D55E00', axis=0) \
                .highlight_max(subset=["ACC ↑", "MSSS ↑"], color='#0072B2', axis=0)
            
            st.dataframe(styled_metrics, use_container_width=True)
        else:
            st.info("Insufficient overlapping valid data points to calculate error metrics.")
            
    else:
        st.warning("Data for the selected variable is missing from the dataset. Please select another variable.")

        
        # ==========================================
# NEW TAB 8: Time Series Decomposition
# ==========================================
with tab8:
    st.header("Time Series Decomposition")
    st.markdown("Meteorological data is inherently cyclical. This algorithm decomposes the raw time series into its underlying **Trend**, **Seasonal cycle**, and the remaining random **Residuals** (where extreme anomaly events like Dzuds often appear).")
    
    decomp_var = st.selectbox("Select variable to decompose:", atmospheric_vars, format_func=get_long_name, key="decomp_var")
    
    if time_col:
        # Group by time and calculate the mean to get a continuous basin-wide signal
        decomp_df = filtered_df.groupby(time_col)[decomp_var].mean().reset_index()
        decomp_df = decomp_df.set_index(time_col)
        
        # Ensure a continuous frequency and interpolate any missing gaps
        # Assuming yearly data for this specific dataset structure
        decomp_df = decomp_df.resample('YE').mean().interpolate(method='linear')
        
        if len(decomp_df) >= 4: # Requires enough data points to detect a cycle
            try:
                from statsmodels.tsa.seasonal import seasonal_decompose
                
                # Perform decomposition
                decomposition = seasonal_decompose(decomp_df[decomp_var], model='additive', period=3) # Period adjustable based on temporal resolution
                
                # Create a multi-plot figure using Plotly Graph Objects
                from plotly.subplots import make_subplots
                fig_decomp = make_subplots(rows=4, cols=1, shared_xaxes=True, 
                                           subplot_titles=("Raw Data", "Underlying Trend", "Seasonality", "Residuals (Anomalies)"))
                
                # 1. Observed
                fig_decomp.add_trace(go.Scatter(x=decomp_df.index, y=decomposition.observed, mode='lines', name='Observed', line=dict(color='#0072B2')), row=1, col=1)
                # 2. Trend
                fig_decomp.add_trace(go.Scatter(x=decomp_df.index, y=decomposition.trend, mode='lines', name='Trend', line=dict(color='#D55E00')), row=2, col=1)
                # 3. Seasonal
                fig_decomp.add_trace(go.Scatter(x=decomp_df.index, y=decomposition.seasonal, mode='lines', name='Seasonal', line=dict(color='#009E73')), row=3, col=1)
                # 4. Residual
                fig_decomp.add_trace(go.Scatter(x=decomp_df.index, y=decomposition.resid, mode='markers', name='Residual', marker=dict(color='#CC79A7')), row=4, col=1)
                fig_decomp.add_hline(y=0, line_dash="dash", line_color="white", row=4, col=1) # Baseline for residuals
                
                fig_decomp.update_layout(height=800, showlegend=False, margin=dict(t=40, b=40, l=40, r=40))
                st.plotly_chart(fig_decomp, use_container_width=True)
                
            except ImportError:
                st.error("Please install the 'statsmodels' library to use the decomposition feature (`pip install statsmodels`).")
            except ValueError as e:
                st.warning(f"Could not decompose the time series: Not enough continuous data points across the selected year range. Try expanding the Global Year Filter in the sidebar.")
        else:
            st.warning("Please select a wider year range in the sidebar to perform time series decomposition (requires at least 4 continuous periods).")
    else:
        st.warning("No valid time column found for decomposition.")