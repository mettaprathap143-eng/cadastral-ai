import streamlit as st
import geopandas as gpd
import folium
from streamlit_folium import st_folium

st.set_page_config(page_title="CadastralAI", layout="wide", page_icon="🗺️")

@st.cache_data
def load(path):
    try:
        return gpd.read_file(path).to_crs(4326)
    except Exception:
        return gpd.GeoDataFrame()

parcels   = load("parcels_final.geojson")
buildings = load("buildings_clean.geojson")
overlaps  = load("flagged_overlaps.geojson")

st.sidebar.title("🛠️ Layer Control")
show_par = st.sidebar.checkbox("Preliminary parcels", True)
show_bld = st.sidebar.checkbox("Building footprints", True)
show_ovl = st.sidebar.checkbox("Flagged overlaps", True)
uses = sorted(parcels.land_use.unique()) if len(parcels) else []
picked = st.sidebar.multiselect("Filter land-use", uses, default=uses)

st.title("🗺️ CadastralAI — Automated Urban Parcel Mapping")
st.caption("AI feature extraction • Preliminary parcels • Topology validation • Ground Truthing support")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Parcels", len(parcels))
c2.metric("Buildings", len(buildings))
c3.metric("Flagged for review", len(overlaps))
c4.metric("Total area", f"{parcels.area_m2.sum()/10000:.1f} ha" if len(parcels) else "—")

m = folium.Map(location=[28.6506, 77.2303], zoom_start=17,
    tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
    attr="Esri World Imagery")

if show_par and len(parcels):
    view = parcels[parcels.land_use.isin(picked)] if picked else parcels
    folium.GeoJson(view, style_function=lambda f: {"color": "#ff7800", "weight": 1, "fillOpacity": 0.05},
        tooltip=folium.GeoJsonTooltip(fields=["parcel_id", "land_use", "area_m2"],
        aliases=["Parcel", "Land use", "Area (m²)"])).add_to(m)
if show_bld and len(buildings):
    folium.GeoJson(buildings, style_function=lambda f: {"color": "#00b3ff", "weight": 1, "fillOpacity": 0.2}).add_to(m)
if show_ovl and len(overlaps):
    folium.GeoJson(overlaps, style_function=lambda f: {"color": "#ff0000", "weight": 2, "fillOpacity": 0.5}).add_to(m)

st_folium(m, height=520, use_container_width=True)

st.subheader("Land-use classification")
if len(parcels):
    st.bar_chart(parcels["land_use"].value_counts())

st.subheader("⬇️ Download GIS-ready outputs")
if len(parcels):
    st.download_button("Parcels (GeoJSON)", parcels.to_json(), "parcels_final.geojson")
if len(buildings):
    st.download_button("Buildings (GeoJSON)", buildings.to_json(), "buildings_clean.geojson")
