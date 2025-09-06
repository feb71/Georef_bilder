
import os, glob, shutil, io, zipfile
import streamlit as st
import pandas as pd
from PIL import Image
import piexif
from pyproj import Transformer, CRS

st.set_page_config(page_title="Geotagging bilder • NTM/UTM → EXIF (WGS84)", layout="wide")

# ===== Helpers =====
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

def ensure_epsg_block(title: str, key_base: str, default: int = 25832):
    st.markdown(f"**{title}**")
    presets = {
        "EUREF89 / UTM32 (EPSG:25832)": 25832,
        "EUREF89 / UTM33 (EPSG:25833)": 25833,
        "NTM10–23 (skriv EPSG manuelt)": None,
        "WGS84 (EPSG:4326)": 4326,
        "Custom EPSG": None,
    }
    label = st.selectbox("Velg standard EPSG (eller «Custom EPSG» og skriv inn under):",
                         list(presets.keys()), index=0, key=f"{key_base}_select")
    code = presets[label]
    custom = st.text_input("Custom EPSG (kun tall, f.eks. 5118 for NTM18):",
                           value="", key=f"{key_base}_custom") if code is None else ""
    epsg = code if code is not None else (int(custom) if custom.strip().isdigit() else None)
    if epsg is None:
        st.info(f"Ingen EPSG valgt – bruker default {default}.")
        epsg = default
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

def list_subdirs(path):
    try:
        return [d for d in sorted(os.listdir(path)) if os.path.isdir(os.path.join(path, d))]
    except Exception:
        return []

def load_points_table(uploaded):
    if uploaded is None:
        return None
    name = uploaded.name.lower()
    if name.endswith(".xlsx") or name.endswith(".xls"):
        try:
            df = pd.read_excel(uploaded)
        except Exception as e:
            st.error(f"Kunne ikke lese Excel: {e}")
            return None
    else:
        try:
            df = pd.read_csv(uploaded)
        except Exception as e:
            st.error(f"Kunne ikke lese CSV: {e}")
            return None
    # normaliser kolonnenavn
    df.columns = [c.strip().lower() for c in df.columns]
    # Forvent minst x/y; epsg valgfritt; id/navn valgfritt
    if not {"x","y"}.issubset(set(df.columns)):
        st.error("Punktliste må minst ha kolonnene: x, y. (epsg, navn/id er valgfritt)")
        return None
    # fallbacks
    if "epsg" not in df.columns:
        df["epsg"] = None
    if "navn" not in df.columns and "id" not in df.columns:
        df["navn"] = [f"Punkt {i+1}" for i in range(len(df))]
    return df

# ===== UI =====
st.title("Geotagging av bilder (NTM/UTM → EXIF WGS84)")

st.info("**EXIF lagrer alltid WGS84.** I appen velger du *inn-CRS* for X/Y (NTM/UTM). "
        "CSV-«Dokumentasjons-CRS» gjelder kun for eksportert liste – ikke EXIF.")

tab_mappe, tab_csv = st.tabs(["Mappe/ZIP med ett punkt", "CSV-mapping for mange bilder"])

with tab_mappe:
    st.subheader("A) Velg bilder + ett punkt (fra liste eller manuelt)")

    mode = st.radio("Kjøring:", ["Lokal disk (mappe)", "Opplasting (ZIP)"], index=0, key="A_mode")
    colA1, colA2 = st.columns(2)
    with colA1:
        if mode == "Lokal disk (mappe)":
            root = st.text_input("Start-rot (lokal maskin/server)", value="", key="A_root")
            cur = st.text_input("Aktiv mappe (kan lime inn eller navigere under)", value="", key="A_cur")
            if root and not cur:
                cur = root
                st.session_state["A_cur"] = cur
            if cur:
                subs = list_subdirs(cur)
                if subs:
                    sel = st.selectbox("Undermapper:", ["(bruk aktiv mappe)"] + subs, key="A_subs")
                    if sel != "(bruk aktiv mappe)":
                        new = os.path.join(cur, sel)
                        st.write(f"Ny aktiv mappe: `{new}`")
                        st.session_state["A_cur"] = new
                        cur = new
                st.code(cur, language="text")
            in_folder = cur
            out_folder = st.text_input("Ut-mappe (kopier dit; tomt = overskriv i aktiv mappe)", value="", key="A_out")
            overwrite = st.checkbox("Overskriv originaler i aktiv mappe (ikke anbefalt)", value=False, key="A_overwrite")
        else:
            zip_up = st.file_uploader("Last opp ZIP med JPG-bilder", type=["zip"], key="A_zip")
            in_folder = None
            out_folder = None
            overwrite = False

    with colA2:
        st.markdown("#### Velg punkt (unngå tasting)")
        pts_up = st.file_uploader("Last opp **punktliste** (Excel/CSV med minst x,y,[epsg],[navn/id])", type=["xlsx","xls","csv"], key="A_pts")
        pts_df = load_points_table(pts_up)
        epsg_in = None
        if pts_df is not None:
            show_cols = [c for c in ["navn","id","x","y","epsg"] if c in pts_df.columns]
            st.dataframe(pts_df[show_cols])
            options = pts_df.index.tolist()
            idx = st.selectbox("Velg punkt:", options, format_func=lambda i: f"{pts_df.iloc[i].get('navn', pts_df.iloc[i].get('id', f'#{i+1}'))}  (x={pts_df.iloc[i]['x']}, y={pts_df.iloc[i]['y']}, epsg={pts_df.iloc[i].get('epsg')})", key="A_pick")
            x = str(pts_df.iloc[idx]["x"])
            y = str(pts_df.iloc[idx]["y"])
            epsg_in = int(pts_df.iloc[idx]["epsg"]) if pd.notna(pts_df.iloc[idx]["epsg"]) else None
        else:
            st.markdown("#### Eller fyll inn manuelt")
            x = st.text_input("X Øst (inn-CRS)", "", key="A_x")
            y = st.text_input("Y Nord (inn-CRS)", "", key="A_y")

        if epsg_in is None:
            epsg_in = ensure_epsg_block("Inn-CRS (NTM/UTM) for valgte X/Y", key_base="A_epsg_in", default=25832)

        epsg_out_doc = ensure_epsg_block("Dokumentasjons-CRS for CSV (kun eksport, ikke EXIF)", key_base="A_epsg_doc", default=25832)
        alt = st.text_input("Høyde (valgfri, meter)", "", key="A_alt")
        direction = st.text_input("Retning (valgfri, grader 0–360)", "", key="A_dir")

    runA = st.button("Kjør geotag (denne seksjonen)", key="A_run")
    if runA:
        try:
            if mode == "Lokal disk (mappe)":
                if not in_folder or not os.path.isdir(in_folder):
                    st.error("Aktiv mappe finnes ikke.")
                else:
                    lat, lon = transform_xy_to_wgs84(float(x), float(y), epsg_in)
                    rows = []
                    jpgs = sorted(glob.glob(os.path.join(in_folder, "*.jpg")) + glob.glob(os.path.join(in_folder, "*.JPG")))
                    if not jpgs:
                        st.warning("Fant ingen JPG i aktiv mappe.")
                    out_dir = in_folder if overwrite or not out_folder else out_folder
                    if not overwrite and out_folder:
                        os.makedirs(out_folder, exist_ok=True)
                    for p in jpgs:
                        fname = os.path.basename(p)
                        dst = p if overwrite or not out_folder else os.path.join(out_folder, fname)
                        if (not overwrite) and out_folder and (p != dst):
                            shutil.copy2(p, dst)
                        write_exif(dst, dst, lat, lon, alt if alt else None, direction if direction else None)
                        Xdoc, Ydoc = transform_xy_to_epsg(float(x), float(y), epsg_in, epsg_out_doc)
                        rows.append({"file": fname, "x_in": x, "y_in": y, "lat": lat, "lon": lon,
                                     f"X_{epsg_out_doc}": Xdoc, f"Y_{epsg_out_doc}": Ydoc, "alt": alt, "dir": direction})
                    df = pd.DataFrame(rows)
                    st.success(f"Geotagget {len(df)} bilder i: {out_dir}")
                    st.download_button("Last ned CSV med posisjoner", df.to_csv(index=False).encode("utf-8"),
                                       "geotag_mappe.csv", "text/csv", key="A_csv")

            else:  # ZIP upload mode
                if zip_up is None:
                    st.error("Last opp en ZIP med bilder.")
                else:
                    lat, lon = transform_xy_to_wgs84(float(x), float(y), epsg_in)
                    rows = []
                    in_mem = io.BytesIO(zip_up.read())
                    zin = zipfile.ZipFile(in_mem, "r")
                    zout_mem = io.BytesIO()
                    zout = zipfile.ZipFile(zout_mem, "w", zipfile.ZIP_DEFLATED)

                    tempdir = "tmp_zip_in"
                    os.makedirs(tempdir, exist_ok=True)
                    for name in zin.namelist():
                        if not name.lower().endswith((".jpg", ".jpeg")):  # prosesser kun bilder
                            continue
                        src_bytes = zin.read(name)
                        # skriv midlertidig fil
                        tmp_path = os.path.join(tempdir, os.path.basename(name))
                        with open(tmp_path, "wb") as f:
                            f.write(src_bytes)
                        # geotagg
                        write_exif(tmp_path, tmp_path, lat, lon, alt if alt else None, direction if direction else None)
                        # legg tilbake i ny zip
                        with open(tmp_path, "rb") as f:
                            zout.writestr(name, f.read())
                        Xdoc, Ydoc = transform_xy_to_epsg(float(x), float(y), epsg_in, epsg_out_doc)
                        rows.append({"file": name, "x_in": x, "y_in": y, "lat": lat, "lon": lon,
                                     f"X_{epsg_out_doc}": Xdoc, f"Y_{epsg_out_doc}": Ydoc, "alt": alt, "dir": direction})
                    zin.close(); zout.close()
                    df = pd.DataFrame(rows)
                    st.success(f"Geotagget {len(df)} bilder i opplastet ZIP.")
                    st.download_button("Last ned ZIP med geotaggede bilder", data=zout_mem.getvalue(),
                                       file_name="geotagged.zip", mime="application/zip", key="A_zip_dl")
                    st.download_button("Last ned CSV med posisjoner", df.to_csv(index=False).encode("utf-8"),
                                       "geotag_zip.csv", "text/csv", key="A_zip_csv")
        except Exception as e:
            st.exception(e)

with tab_csv:
    st.subheader("B) CSV-mapping for mange bilder (lokal disk)")
    st.markdown("To CSV-varianter støttes i dag (bruk én av dem):\n"
                "- **Folder-form**: `folder,x,y,alt,dir[,epsg]` → alle JPG i `root/folder` får samme posisjon\n"
                "- **File-form**:   `file,x,y,alt,dir[,epsg]`   → én rad per fil (sti relativt til root)\n"
                "Hvis `epsg` mangler i CSV, bruk velgeren under for inn-CRS.\n")

    colB1, colB2 = st.columns(2)
    with colB1:
        root = st.text_input("Root-mappe (lokal)", value="", key="B_root")
        csv_up = st.file_uploader("Last opp mapping-CSV", type=["csv"], key="B_csv")
        out_root = st.text_input("Ut-root (skriv kopier)", value="", key="B_out")
        overwrite2 = st.checkbox("Overskriv originale filer", value=False, key="B_overwrite")
    with colB2:
        epsg_in2_default = ensure_epsg_block("Inn-CRS standard (brukes hvis CSV-rad mangler epsg)", key_base="B_epsg_in", default=25832)
        epsg_out_doc2 = ensure_epsg_block("Dokumentasjons-CRS for CSV (kun eksport)", key_base="B_epsg_doc", default=25832)

    runB = st.button("Kjør geotag (CSV)", key="B_run")
    if runB:
        try:
            if not root or not os.path.isdir(root):
                st.error("Root-mappe finnes ikke.")
            elif not csv_up:
                st.error("Last opp en CSV.")
            else:
                import csv as _csv
                dfm = pd.read_csv(csv_up)
                rows = []
                os.makedirs(out_root, exist_ok=True) if (out_root and not overwrite2) else None

                def row_epsg(row):
                    if "epsg" in dfm.columns and not pd.isna(row.get("epsg", None)):
                        try:
                            return int(row["epsg"])
                        except:
                            return epsg_in2_default
                    return epsg_in2_default

                def process_target(path_list, X, Y, Alt, Dir, epsg_row):
                    lat, lon = transform_xy_to_wgs84(X, Y, epsg_row)
                    for p in path_list:
                        if not os.path.isfile(p):
                            st.warning(f"Mangler fil: {p}")
                            continue
                        fname = os.path.relpath(p, root)
                        dst = p if overwrite2 else os.path.join(out_root, fname) if out_root else p
                        os.makedirs(os.path.dirname(dst), exist_ok=True)
                        if (not overwrite2) and (p != dst):
                            shutil.copy2(p, dst)
                        write_exif(dst, dst, lat, lon, Alt if pd.notna(Alt) else None, Dir if pd.notna(Dir) else None)
                        Xdoc, Ydoc = transform_xy_to_epsg(X, Y, epsg_row, epsg_out_doc2)
                        rows.append({"file": fname, "x_in": X, "y_in": Y, "epsg_in": epsg_row,
                                     "lat": lat, "lon": lon, f"X_{epsg_out_doc2}": Xdoc, f"Y_{epsg_out_doc2}": Ydoc,
                                     "alt": Alt, "dir": Dir})

                # Folder-form
                if {"folder","x","y"}.issubset(dfm.columns):
                    for _, r in dfm.dropna(subset=["folder","x","y"]).iterrows():
                        folder = os.path.join(root, str(r["folder"]))
                        jpgs = sorted(glob.glob(os.path.join(folder, "*.jpg")) + glob.glob(os.path.join(folder, "*.JPG")))
                        process_target(jpgs, float(r["x"]), float(r["y"]), r.get("alt", None), r.get("dir", None), row_epsg(r))

                # File-form
                if {"file","x","y"}.issubset(dfm.columns):
                    for _, r in dfm.dropna(subset=["file","x","y"]).iterrows():
                        fpath = os.path.join(root, str(r["file"]))
                        process_target([fpath], float(r["x"]), float(r["y"]), r.get("alt", None), r.get("dir", None), row_epsg(r))

                dfr = pd.DataFrame(rows)
                st.success(f"Geotagget {len(dfr)} bilder.")
                st.download_button("Last ned CSV med posisjoner", dfr.to_csv(index=False).encode("utf-8"),
                                   "geotag_csv_mapping.csv", "text/csv", key="B_csv_dl")
        except Exception as e:
            st.exception(e)

st.markdown("---")
st.caption("EXIF er alltid WGS84 (EPSG:4326). Velg inn-CRS riktig for X/Y (NTM/UTM). "
           "Det siste valget av CRS i UI gjelder kun for eksportert CSV (dokumentasjon).")
