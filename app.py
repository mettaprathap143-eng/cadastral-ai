import io, os, itertools, tempfile
import numpy as np
import pandas as pd
import geopandas as gpd
import streamlit as st
import folium
from streamlit_folium import st_folium
import cv2
import rasterio
from shapely.geometry import MultiPoint, box, Polygon
from shapely.ops import voronoi_diagram

st.set_page_config(page_title="CadastralAI", layout="wide", page_icon="🗺️")

ESRI = "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"

# ---------- GPU check: SAM only possible with GPU ----------
try:
    import torch
    GPU = torch.cuda.is_available()
except Exception:
    GPU = False

# ---------- helper functions ----------
def classic_polygons(rgb, min_area_px=40):
    if rgb.ndim == 2:
        gray = rgb
    else:
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
    block = max(31, (min(gray.shape) // 20) | 1)
    th = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY_INV, block, 5)
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for c in contours:
        if cv2.contourArea(c) < min_area_px:
            continue
        pts = cv2.approxPolyDP(c, 1.5, True).reshape(-1, 2)
        if len(pts) >= 3:
            out.append(pts)
    return out

@st.cache_resource
def get_sam():
    from samgeo import SamGeo
    return SamGeo(model_type="vit_b",
                  checkpoint="https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth")

def sam_polygons(tif_path):
    sam = get_sam()
    masks = tempfile.mktemp(suffix="_masks.tif")
    vec = tempfile.mktemp(suffix="_objects.geojson")
    sam.generate(tif_path, masks)
    sam.tiff_to_vector(masks, vec)
    return gpd.read_file(vec)

def build_outputs(b):
    b = b[b.area_m2 > 1].copy()
    b["geometry"] = b.geometry.buffer(0)
    b = b.reset_index(drop=True)
    empty = gpd.GeoDataFrame(geometry=[], crs=b.crs)
    if len(b) < 2:
        return b, empty, empty

    region = box(*b.total_bounds)
    cells = voronoi_diagram(MultiPoint(b.geometry.centroid.tolist()), envelope=region)
    p = gpd.GeoDataFrame(geometry=[c.intersection(region) for c in cells.geoms], crs=b.crs)
    p["parcel_id"] = [f"P{i+1:03d}" for i in range(len(p))]
    p["area_m2"] = p.geometry.area.round(1)

    joined = gpd.sjoin(b[["area_m2", "geometry"]], p[["parcel_id", "geometry"]],
                       predicate="intersects", how="inner")
    stats = joined.groupby("parcel_id").agg(buildings=("area_m2", "count"),
                                            bldg_area=("area_m2", "sum")).reset_index()
    p = p.merge(stats, on="parcel_id", how="left").fillna({"buildings": 0, "bldg_area": 0})
    p["coverage"] = p.bldg_area / p.area_m2
    q1, q2 = p.coverage.quantile([0.33, 0.66])
    p["land_use"] = p.coverage.apply(
        lambda c: "Dense built-up" if c >= q2 else ("Mixed/medium" if c >= q1 else "Low-density/open"))

    cut = b.area_m2.quantile(0.25)
    flags = gpd.GeoDataFrame(
        [{"geometry": r.geometry, "issue": f"small structure ({r.area_m2:.1f} m²) — field check"}
         for _, r in b[b.area_m2 <= cut].iterrows()], crs=b.crs)
    return b, p, flags

@st.cache_data
def load(path):
    try:
        return gpd.read_file(path).to_crs(4326)
    except Exception:
        return gpd.GeoDataFrame()

# ---------- UI ----------
st.title("🗺️ CadastralAI — Automated Urban Parcel Mapping")
st.caption("Upload drone orthomosaic → feature extraction → preliminary parcels → topology flags → GIS-ready GeoJSON")

mode = st.sidebar.radio("Data source",
                        ["📂 Demo area (Chandni Chowk)", "🛰️ Upload your drone image"])

# ============ MODE 1: DEMO DATA ============
if mode == "📂 Demo area (Chandni Chowk)":
    parcels   = load("parcels_final.geojson")
    buildings = load("buildings_clean.geojson")
    overlaps  = load("flagged_overlaps.geojson")

    st.sidebar.title("🛠️ Layer Control")
    show_par = st.sidebar.checkbox("Preliminary parcels", True)
    show_bld = st.sidebar.checkbox("Building footprints", True)
    show_ovl = st.sidebar.checkbox("Flagged for review", True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Parcels", len(parcels))
    c2.metric("Buildings", len(buildings))
    c3.metric("Flagged", len(overlaps))
    c4.metric("Total area", f"{parcels.area_m2.sum()/10000:.1f} ha" if len(parcels) else "—")

    m = folium.Map(location=[28.6506, 77.2303], zoom_start=17, tiles=ESRI, attr="Esri World Imagery")
    if show_par and len(parcels):
        folium.GeoJson(parcels, style_function=lambda f: {"color": "#ff7800", "weight": 1, "fillOpacity": 0.05},
            tooltip=folium.GeoJsonTooltip(fields=["parcel_id", "land_use", "area_m2"],
            aliases=["Parcel", "Land use", "Area (m²)"])).add_to(m)
    if show_bld and len(buildings):
        folium.GeoJson(buildings, style_function=lambda f: {"color": "#00b3ff", "weight": 1, "fillOpacity": 0.2}).add_to(m)
    if show_ovl and len(overlaps):
        folium.GeoJson(overlaps, style_function=lambda f: {"color": "#ff0000", "weight": 2, "fillOpacity": 0.5}).add_to(m)
    st_folium(m, height=500, use_container_width=True)

    d1, d2, d3 = st.columns(3)
    if len(parcels):   d1.download_button("⬇️ Parcels (GeoJSON)", parcels.to_json(), "parcels_final.geojson")
    if len(buildings): d2.download_button("⬇️ Buildings (GeoJSON)", buildings.to_json(), "buildings_clean.geojson")
    if len(overlaps):  d3.download_button("⬇️ Flags (GeoJSON)", overlaps.to_json(), "flagged_overlaps.geojson")

# ============ MODE 2: UPLOAD YOUR OWN ============
else:
    st.sidebar.write("Engine: " + ("🧠 SAM (GPU detected)" if GPU else "⚡ Fast CV (cloud)"))
    gsd = st.sidebar.number_input("Pixel size, metres (used if image lacks coordinates)",
                                  0.005, 2.0, 0.05, 0.005, format="%.3f")

    up = st.file_uploader("Drop your drone orthomosaic (.tif / .tiff — JPG/PNG also work)",
                          type=["tif", "tiff", "png", "jpg", "jpeg"])

    if up is not None:
        bytes_ = up.getvalue()
        crs, transform, georef = None, None, False
        rgb = None
        try:
            src = rasterio.open(io.BytesIO(bytes_))
            scale = max(1, src.width // 2000)
            W, H = src.width // scale, src.height // scale
            data = src.read(out_shape=(src.count, H, W))
            arr = data[:3] if src.count >= 3 else np.repeat(data[:1], 3, axis=0)
            if arr.shape[0] > 3:
                arr = arr[:3]
            if arr.shape[0] < 3:
                arr = np.repeat(arr[:1], 3, axis=0)
            lo, hi = np.percentile(arr, [2, 98])
            arr = np.clip((arr.astype("float32") - lo) / max(hi - lo, 1) * 255, 0, 255).astype("uint8")
            rgb = np.transpose(arr, (1, 2, 0))   # rasterio (bands,H,W) → OpenCV (H,W,bands)
            transform = src.transform * src.transform.scale(scale, scale)
            crs = src.crs
            georef = crs is not None
            st.sidebar.success(f"{src.width}×{src.height} → working {W}×{H} | CRS: {crs}")
        except Exception:
            try:
                from PIL import Image
                img = Image.open(io.BytesIO(bytes_)).convert("RGB")
                scale = max(1, img.width // 2000)
                img = img.resize((img.width // scale, img.height // scale))
                rgb = np.array(img)
                st.sidebar.info("Read as plain image (no coordinates).")
            except Exception as e:
                st.error(f"Could not read this file: {e}")

        if rgb is not None:
            use_sam = GPU and st.sidebar.checkbox("🧠 Use SAM AI (GPU)", True)

            if st.button("⚙️ Run AI extraction", type="primary"):
                with st.spinner("Extracting features…"):
                    try:
                        if use_sam:
                            tif = tempfile.mktemp(suffix="_in.tif")
                            with rasterio.open(tif, "w", driver="GTiff", width=rgb.shape[1],
                                               height=rgb.shape[0], count=3, dtype="uint8",
                                               crs=(crs if georef else "EPSG:4326"),
                                               transform=(transform if georef else
                                                          rasterio.transform.from_origin(0, 0, 1e-5, 1e-5))) as dst:
                                dst.write(rgb.transpose(2, 0, 1))
                            raw = sam_polygons(tif)
                            if raw.crs is not None and raw.crs.is_geographic:
                                raw = raw.to_crs(raw.estimate_utm_crs())
                        else:
                            polys = classic_polygons(rgb)
                            if georef:
                                recs = []
                                for pts in polys:
                                    wx, wy = rasterio.transform.xy(transform,
                                                                   pts[:, 1].tolist(), pts[:, 0].tolist())
                                    recs.append(Polygon(zip(wx, wy)))
                                raw = gpd.GeoDataFrame(geometry=recs, crs=crs)
                                if raw.crs.is_geographic:
                                    raw = raw.to_crs(raw.estimate_utm_crs())
                            else:
                                recs = [Polygon(pts.astype(float) * gsd) for pts in polys]
                                raw = gpd.GeoDataFrame(geometry=recs)
                        raw["area_m2"] = raw.geometry.area
                        b, p, flags = build_outputs(raw)
                        st.session_state["res"] = (b, p, flags, georef, rgb, gsd)
                    except Exception as e:
                        st.error(f"Processing failed: {e}")

            if "res" in st.session_state:
                b, p, flags, georef, rgb, gsd = st.session_state["res"]
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Footprints", len(b))
                c2.metric("Parcels", len(p))
                c3.metric("Flagged", len(flags))
                c4.metric("Built-up", f"{b.area_m2.sum()/10000:.2f} ha" if len(b) else "—")

                if georef and len(b):
                    bb = b.to_crs(4326).total_bounds
                    m = folium.Map(location=[(bb[1] + bb[3]) / 2, (bb[0] + bb[2]) / 2],
                                   zoom_start=17, tiles=ESRI, attr="Esri World Imagery")
                    folium.GeoJson(b.to_crs(4326), style_function=lambda f: {"color": "#00b3ff", "weight": 1, "fillOpacity": 0.25}).add_to(m)
                    if len(p):
                        folium.GeoJson(p.to_crs(4326), style_function=lambda f: {"color": "#ff7800", "weight": 1, "fillOpacity": 0.05},
                            tooltip=folium.GeoJsonTooltip(fields=["parcel_id", "land_use", "area_m2"])).add_to(m)
                    if len(flags):
                        folium.GeoJson(flags.to_crs(4326), style_function=lambda f: {"color": "#ff0000", "weight": 2, "fillOpacity": 0.5}).add_to(m)
                    st_folium(m, height=500, use_container_width=True)
                elif len(b):
                    st.info("This image has no coordinates — overlay view (areas estimated from pixel size).")
                    canvas = rgb.copy()
                    for poly in b.geometry:
                        pts = (np.array(poly.exterior.coords) / gsd).astype(np.int32)
                        cv2.polylines(canvas, [pts], True, (0, 179, 255), 2)
                    st.image(canvas)

                d1, d2, d3 = st.columns(3)
                d1.download_button("⬇️ parcels_final.geojson", p.to_json(), "parcels_final.geojson")
                d2.download_button("⬇️ buildings_clean.geojson", b.to_json(), "buildings_clean.geojson")
                d3.download_button("⬇️ flagged_overlaps.geojson", flags.to_json(), "flagged_overlaps.geojson")
    else:
        st.info("⬆️ Upload a drone orthomosaic to begin — or switch to the demo area in the sidebar.")
