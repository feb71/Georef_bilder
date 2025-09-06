
import os, glob, shutil
import streamlit as st
import pandas as pd
from PIL import Image
import piexif
from pyproj import Transformer, CRS

st.set_page_config(page_title="Foto-geotagging (NTM/UTM → EXIF WGS84)", layout="wide")

# ---------- helpers ----------
def deg_to_dms_rational(dd):
    sign = 1 if dd >= 0 else -1
    dd = abs(dd)
    d = int(dd)
    m_full = (dd - d) * 60
    m = int(m_full)
    s = round((m_full - m) * 60 * 10000)
    return sign, ((d,1),(m,1),(s,10000))

def exif_gps(lat_dd, lon_dd, alt=None, direction=None):
    sign_lat, lat = deg_to_dms_rational(lat_dd)
    sign_lon, lon = deg_to_dms_rational(lon_dd)
    gps = {
        piexif.GPSIFD.GPSLatitudeRef: b'N' if sign_lat>=0 else b'S',
        piexif.GPSIFD.GPSLatitude: lat,
        piexif.GPSIFD.GPSLongitudeRef: b'E' if sign_lon>=0 else b'W',
        piexif.GPSIFD.GPSLongitude: lon,
    }
    if alt is not None and str(alt) != "":
        alt = float(alt)
        gps[piexif.GPSIFD.GPSAltitudeRef] = 0 if alt >= 0 else 1
        gps[piexif.GPSIFD.GPSAltitude] = (int(round(abs(alt)*100)), 100)
    if direction is not None and str(direction) != "":
        direction = float(direction)
        gps[piexif.GPSIFD.GPSImgDirectionRef] = b'T'  # true north
        gps[piexif.GPSIFD.GPSImgDirection] = (int(round(direction*100)), 100)
    return gps

def write_exif(path_in, path_out, lat, lon, alt=None, direction=None):
    im = Image.open(path_in)
    try:
        exif_dict = piexif.load(im.info.get("exif", b""))
    except Exception:
        exif_dict = {"0th":{}, "Exif":{}, "GPS":{}, "1st":{}}
    exif_dict["GPS"] = exif_gps(lat, lon, alt, direction)
    exif_bytes = piexif.dump(exif_dict)
    im.save(path_out, "jpeg", exif=exif_bytes)

def pick_epsg_label(key_base: str, default_index: int = 0):
    """
    Keyed selectbox + optional custom text_input to avoid duplicate element IDs.
    """
    presets = {
        "EUREF89 / UTM32 (EPSG:25832)": 25832,
        "EUREF89 / UTM33 (EPSG:25833)": 25833,
        "NTM10–23 (skriv EPSG manuelt)": None,
        "WGS84 (EPSG:4326)": 4326,
        "Custom EPSG": None,
    }
    label = st.selectbox(
        "Velg standard EPSG (eller velg «Custom EPSG» og skriv inn under):",
        list(presets.keys()),
        index=default_index,
        key=f"{key_base}_selectbox",
    )
    code = presets[label]
    custom = st.text_input(
        "Custom EPSG (kun tall, f.eks. 5118 for NTM18):",
        value="",
        key=f"{key_base}_custom_epsg",
    ) if code is None else ""
    epsg = None
    if code is not None:
        epsg = code
    elif custom.strip().isdigit():
        epsg = int(custom.strip())
    return epsg

def ensure_epsg(name: str, default: int, key_base: str):
    st.markdown(f"**{name}**")
    epsg = pick_epsg_label(key_base=key_base)
    if epsg is None:
        st.info(f"Ingen EPSG valgt – bruker default {default}.")
        epsg = default
    # Validate
    try:
        _ = CRS.from_epsg(epsg)
    except Exception:
        st.error(f"Ugyldig EPSG: {epsg}. Bruker default {default}.")
        epsg = default
    return epsg

def transform_xy_to_wgs84(x, y, src_epsg):
    tr = Transformer.from_crs(src_epsg, 4326, always_xy=True)
    lon, lat = tr.transform(float(x), float(y))
    return lat, lon

def transform_xy_to_epsg(x, y, src_epsg, dst_epsg):
    if src_epsg == dst_epsg:
        return float(x), float(y)
    tr = Transformer.from_crs(src_epsg, dst_epsg, always_xy=True)
    X, Y = tr.transform(float(x), float(y))
    return X, Y

# ---------- UI ----------
st.title("Geotagging av bilder (NTM/UTM → EXIF WGS84)")

tab_mappe, tab_csv = st.tabs(["Samme posisjon for en mappe", "CSV-mapping for mange bilder"])

with tab_mappe:
    st.subheader("A) Sett samme posisjon på alle JPG i en mappe")
    col1, col2 = st.columns(2)
    with col1:
        in_folder = st.text_input("Innmappesti (f.eks. D:\\\\prosjekt\\\\Bilder\\\\KUM_001)", "", key="A_in_folder")
        out_folder = st.text_input("Ut-mappe (skriv til kopier, ikke originaler)", "", key="A_out_folder")
        overwrite = st.checkbox("Overskriv originaler i inn-mappa (ikke anbefalt)", value=False, key="A_overwrite")
    with col2:
        epsg_in = ensure_epsg("Inn-CRS (NTM/UTM)", 25832, key_base="A_epsg_in")
        epsg_out_doc = ensure_epsg("Dokumentasjons-CRS for CSV (f.eks. 25832/25833)", 25832, key_base="A_epsg_doc")

    col3, col4, col5 = st.columns(3)
    with col3:
        x = st.text_input("X Øst (inn-CRS)", "", key="A_x")
        y = st.text_input("Y Nord (inn-CRS)", "", key="A_y")
    with col4:
        alt = st.text_input("Høyde (valgfri, meter)", "", key="A_alt")
        direction = st.text_input("Retning (valgfri, grader 0–360)", "", key="A_dir")
    with col5:
        run1 = st.button("Kjør geotag (mappe)", key="A_run")

    if run1:
        if not in_folder or not os.path.isdir(in_folder):
            st.error("Innmappa finnes ikke.")
        elif (not overwrite) and (not out_folder):
            st.error("Oppgi ut-mappe når du ikke overskriver.")
        elif x.strip()=="" or y.strip()=="":
            st.error("Oppgi X og Y.")
        else:
            try:
                lat, lon = transform_xy_to_wgs84(float(x), float(y), epsg_in)
                st.write(f"WGS84: lat={lat:.6f}, lon={lon:.6f}")
                rows = []
                jpgs = sorted(glob.glob(os.path.join(in_folder, "*.jpg")) + glob.glob(os.path.join(in_folder, "*.JPG")))
                if not jpgs:
                    st.warning("Fant ingen JPG i inn-mappa.")
                os.makedirs(out_folder, exist_ok=True) if out_folder else None
                for p in jpgs:
                    fname = os.path.basename(p)
                    dst = p if overwrite else os.path.join(out_folder, fname)
                    if (not overwrite) and (p != dst):
                        os.makedirs(os.path.dirname(dst), exist_ok=True)
                        shutil.copy2(p, dst)
                    write_exif(dst, dst, lat, lon, alt if alt else None, direction if direction else None)
                    Xdoc, Ydoc = transform_xy_to_epsg(float(x), float(y), epsg_in, epsg_out_doc)
                    rows.append({"file": fname, "in_folder": in_folder, "x_in": x, "y_in": y,
                                 "lat": lat, "lon": lon, f"X_{epsg_out_doc}": Xdoc, f"Y_{epsg_out_doc}": Ydoc})
                df = pd.DataFrame(rows)
                st.success(f"Geotagget {len(df)} bilder.")
                st.download_button("Last ned CSV med posisjoner", df.to_csv(index=False).encode("utf-8"),
                                   "geotag_mappe.csv", "text/csv", key="A_download")
            except Exception as e:
                st.exception(e)

with tab_csv:
    st.subheader("B) CSV-mapping (ulike posisjoner)")
    st.markdown("""
    **CSV-format (ett av to):**
    - *Folder-form*: `folder,x,y,alt,dir`  → alle JPG i `root/folder` får samme posisjon  
    - *File-form*:   `file,x,y,alt,dir`    → én rad per bilde (sti relativt til root)
    """)
    col1, col2 = st.columns(2)
    with col1:
        root = st.text_input("Root-mappe (inneholder undermapper/bilder)", "", key="B_root")
        csv_file = st.file_uploader("Last opp mapping-CSV", type=["csv"], key="B_csv")
        out_root = st.text_input("Ut-root (skriv kopier)", "", key="B_out_root")
        overwrite2 = st.checkbox("Overskriv originale filer", value=False, key="B_overwrite")
    with col2:
        epsg_in2 = ensure_epsg("Inn-CRS (NTM/UTM)", 25832, key_base="B_epsg_in")
        epsg_out_doc2 = ensure_epsg("Dokumentasjons-CRS (CSV)", 25832, key_base="B_epsg_doc")

    run2 = st.button("Kjør geotag (CSV-mapping)", key="B_run")

    if run2:
        if not root or not os.path.isdir(root):
            st.error("Root-mappe finnes ikke.")
        elif (not overwrite2) and (not out_root):
            st.error("Oppgi ut-root når du ikke overskriver.")
        elif not csv_file:
            st.error("Last opp en CSV.")
        else:
            try:
                dfm = pd.read_csv(csv_file)
                rows = []
                os.makedirs(out_root, exist_ok=True) if out_root else None

                def process_target(path_list, X, Y, Alt, Dir):
                    lat, lon = transform_xy_to_wgs84(X, Y, epsg_in2)
                    for p in path_list:
                        if not os.path.isfile(p):
                            st.warning(f"Mangler fil: {p}")
                            continue
                        fname = os.path.relpath(p, root)
                        dst = p if overwrite2 else os.path.join(out_root, fname)
                        os.makedirs(os.path.dirname(dst), exist_ok=True)
                        if (not overwrite2) and (p != dst):
                            shutil.copy2(p, dst)
                        write_exif(dst, dst, lat, lon, Alt if pd.notna(Alt) else None, Dir if pd.notna(Dir) else None)
                        Xdoc, Ydoc = transform_xy_to_epsg(X, Y, epsg_in2, epsg_out_doc2)
                        rows.append({"file": fname, "x_in": X, "y_in": Y, "lat": lat, "lon": lon,
                                     f"X_{epsg_out_doc2}": Xdoc, f"Y_{epsg_out_doc2}": Ydoc,
                                     "alt": Alt, "dir": Dir})

                # Folder-form
                if {"folder","x","y"}.issubset(dfm.columns):
                    for _, r in dfm.dropna(subset=["folder","x","y"]).iterrows():
                        folder = os.path.join(root, str(r["folder"]))
                        jpgs = sorted(glob.glob(os.path.join(folder, "*.jpg")) + glob.glob(os.path.join(folder, "*.JPG")))
                        process_target(jpgs, float(r["x"]), float(r["y"]), r.get("alt", None), r.get("dir", None))

                # File-form
                if {"file","x","y"}.issubset(dfm.columns):
                    for _, r in dfm.dropna(subset=["file","x","y"]).iterrows():
                        fpath = os.path.join(root, str(r["file"]))
                        process_target([fpath], float(r["x"]), float(r["y"]), r.get("alt", None), r.get("dir", None))

                dfr = pd.DataFrame(rows)
                st.success(f"Geotagget {len(dfr)} bilder.")
                st.download_button("Last ned CSV med posisjoner", dfr.to_csv(index=False).encode("utf-8"),
                                   "geotag_csv.csv", "text/csv", key="B_download")
            except Exception as e:
                st.exception(e)

st.markdown("---")
st.caption("Merk: EXIF lagrer alltid WGS84 (lat/lon). Inn-koordinater i NTM/UTM transformeres automatisk. "
           "Kjør én batch per sone (NTM/UTM) for å unngå blanding.")
