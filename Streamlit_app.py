
import os, glob, shutil, io, zipfile, re, math
import streamlit as st
import pandas as pd
from PIL import Image
import piexif
from pyproj import Transformer, CRS

st.set_page_config(page_title="Geotagging bilder • Gemini-skjema → EXIF WGS84 (med filnavn)", layout="wide")

# ===== Helpers =====
def deg_to_dms_rational(dd):
    sign = 1 if dd >= 0 else -1
    dd = abs(dd)
    d = int(dd)
    m_full = (dd - d) * 60
    m = int(m_full)
    s = round((m_full - m) * 60 * 10000)
    return sign, ((d,1),(m,1),(s,10000))

def _is_valid_number(x):
    try:
        fx = float(x)
    except Exception:
        return False
    return not (math.isnan(fx) or math.isinf(fx))

def _wrap_deg(d):
    d = float(d) % 360.0
    if d < 0:
        d += 360.0
    return d

def exif_gps(lat_dd, lon_dd, alt=None, direction=None):
    sign_lat, lat = deg_to_dms_rational(lat_dd)
    sign_lon, lon = deg_to_dms_rational(lon_dd)
    gps = {
        piexif.GPSIFD.GPSLatitudeRef: b'N' if sign_lat>=0 else b'S',
        piexif.GPSIFD.GPSLatitude: lat,
        piexif.GPSIFD.GPSLongitudeRef: b'E' if sign_lon>=0 else b'W',
        piexif.GPSIFD.GPSLongitude: lon,
    }
    if _is_valid_number(alt):
        alt = float(alt)
        gps[piexif.GPSIFD.GPSAltitudeRef] = 0 if alt >= 0 else 1
        gps[piexif.GPSIFD.GPSAltitude] = (int(round(abs(alt)*100)), 100)
    if _is_valid_number(direction):
        direction = _wrap_deg(direction)
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

def parse_float_maybe_comma(v):
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return float(v)
    s = str(v).strip().replace(" ", "").replace("\xa0","")
    if s == "" or s.lower() in {"nan","none","-"}:
        return None
    if "," in s and "." in s:
        if s.rfind(",") > s.rfind("."):
            s = s.replace(".", "").replace(",", ".")
        else:
            s = s.replace(",", "")
    elif "," in s:
        s = s.replace(",", ".")
    try:
        return float(s)
    except:
        return None

def ensure_epsg(title: str, key_base: str, default: int = 25832):
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

def transform_EN_to_wgs84(E, N, src_epsg):
    tr = Transformer.from_crs(src_epsg, 4326, always_xy=True)
    lon, lat = tr.transform(float(E), float(N))
    return lat, lon

def transform_EN_to_epsg(E, N, src_epsg, dst_epsg):
    if src_epsg == dst_epsg:
        return float(E), float(N)
    tr = Transformer.from_crs(src_epsg, dst_epsg, always_xy=True)
    X, Y = tr.transform(float(E), float(N))
    return X, Y

def list_subdirs(path):
    try:
        return [d for d in sorted(os.listdir(path)) if os.path.isdir(os.path.join(path, d))]
    except Exception:
        return []

def sanitize_for_filename(s):
    if s is None:
        return None
    s = str(s).strip()
    s = re.sub(r'[\\/:*?"<>|]+', '_', s)  # remove illegal chars
    s = re.sub(r'\s+', '_', s)           # spaces to underscore
    s = s.strip('._')                    # avoid leading/trailing dots/underscores
    return s or None

def build_new_name(pattern, label, orig_name, E=None, N=None):
    base, ext = os.path.splitext(orig_name)
    safe_label = sanitize_for_filename(label)
    if pattern == "keep" or not safe_label:
        return orig_name
    if pattern == "label_orig":
        return f"{safe_label}_{base}{ext}"
    if pattern == "label_only":
        return f"{safe_label}{ext}"
    if pattern == "label_en":
        e_txt = f"{int(round(E))}" if E is not None else "E"
        n_txt = f"{int(round(N))}" if N is not None else "N"
        return f"{safe_label}_{e_txt}_{n_txt}{ext}"
    return orig_name

def ensure_unique_path(path):
    if not os.path.exists(path):
        return path
    base, ext = os.path.splitext(path)
    i = 1
    while True:
        candidate = f"{base}_{i}{ext}"
        if not os.path.exists(candidate):
            return candidate
        i += 1

# ===== UI =====
st.title("Geotagging bilder – Gemini-skjema → EXIF (WGS84) + fornuftige filnavn")

st.info("Bruker kolonnene «Øst», «Nord», «Høyde», «Rotasjon», «S_OBJID» hvis de finnes. "
        "EXIF skrives alltid i WGS84. Velg inn-CRS for E/N og mønster for nytt filnavn.")

tab_mappe, tab_csv = st.tabs(["Mappe/ZIP + ett punkt", "CSV/Excel-mapping for mange bilder"])

# ------------- TAB A -------------
with tab_mappe:
    st.subheader("A) Velg bilder + ett punkt (fra fil eller manuelt)")
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

        rename_pattern = st.selectbox("Nytt filnavn (mønster)", [
            "Behold originalt navn",
            "S_OBJID + originalt navn",
            "Kun S_OBJID",
            "S_OBJID + avrundet E/N"
        ], index=1, key="A_rename")

    with colA2:
        st.markdown("#### Velg punkt fra fil (Excel/CSV) – Gemini-skjema")
        pts_up = st.file_uploader("Punktliste (Øst, Nord, Høyde, Rotasjon, S_OBJID)", type=["xlsx","xls","csv"], key="A_pts")
        E = N = Alt = Dir = None
        label = None
        epsg_in = None

        if pts_up is not None:
            name = pts_up.name.lower()
            try:
                if name.endswith((".xlsx",".xls")):
                    df = pd.read_excel(pts_up, dtype=str)
                else:
                    df = pd.read_csv(pts_up, dtype=str)
            except Exception as e:
                st.error(f"Kunne ikke lese punktfil: {e}")
                df = None

            if df is not None and not df.empty:
                low = {c.lower(): c for c in df.columns}
                col_e  = low.get("øst", low.get("oest", low.get("x", None)))
                col_n  = low.get("nord", low.get("y", None))
                col_h  = low.get("høyde", low.get("hoyde", low.get("h", None)))
                col_r  = low.get("rotasjon", low.get("retning", low.get("dir", None)))
                col_id = low.get("s_objid", low.get("navn", low.get("id", None)))

                if not col_e or not col_n:
                    st.error("Fant ikke «Øst»/«Nord» i fila. Sjekk kolonnenavn.")
                else:
                    show_cols = [c for c in [col_id, col_e, col_n, col_h, col_r] if c]
                    st.dataframe(df[show_cols].head(20))

                    idx = st.selectbox("Velg punkt:", df.index.tolist(),
                                       format_func=lambda i: f"{df.loc[i, col_id]}" if col_id else f"Rad {i+1}",
                                       key="A_pick")
                    def pf(v): return parse_float_maybe_comma(v)
                    E   = pf(df.loc[idx, col_e])
                    N   = pf(df.loc[idx, col_n])
                    Alt = pf(df.loc[idx, col_h]) if col_h else None
                    Dir = pf(df.loc[idx, col_r]) if col_r else None
                    label = df.loc[idx, col_id] if col_id else None

        if epsg_in is None:
            epsg_in = ensure_epsg("Inn-CRS (UTM/NTM) for E/N", key_base="A_epsg_in", default=25832)

        epsg_out_doc = ensure_epsg("Dokumentasjons-CRS for CSV (kun eksport, ikke EXIF)", key_base="A_epsg_doc", default=25832)

    runA = st.button("Kjør geotag (denne seksjonen)", key="A_run")
    if runA:
        try:
            if (E is None) or (N is None):
                st.error("E/N mangler eller kunne ikke tolkes som tall.")
            elif mode == "Lokal disk (mappe)":
                if not in_folder or not os.path.isdir(in_folder):
                    st.error("Aktiv mappe finnes ikke.")
                else:
                    lat, lon = transform_EN_to_wgs84(float(E), float(N), epsg_in)
                    rows = []
                    jpgs = sorted(glob.glob(os.path.join(in_folder, "*.jpg")) + glob.glob(os.path.join(in_folder, "*.JPG")))
                    if not jpgs:
                        st.warning("Fant ingen JPG i aktiv mappe.")
                    out_dir = in_folder if overwrite or not out_folder else out_folder
                    if not overwrite and out_folder:
                        os.makedirs(out_folder, exist_ok=True)

                    patt_map = {
                        "Behold originalt navn": "keep",
                        "S_OBJID + originalt navn": "label_orig",
                        "Kun S_OBJID": "label_only",
                        "S_OBJID + avrundet E/N": "label_en",
                    }
                    patt = patt_map[rename_pattern]

                    for p in jpgs:
                        fname = os.path.basename(p)
                        newname = build_new_name(patt, label, fname, E, N)
                        if not overwrite and out_folder:
                            dst = ensure_unique_path(os.path.join(out_folder, newname))
                            shutil.copy2(p, dst)
                        else:
                            dst = os.path.join(os.path.dirname(p), newname) if newname != fname else p
                            if dst != p:
                                dst = ensure_unique_path(dst)
                                shutil.copy2(p, dst)
                        write_exif(dst, dst, lat, lon, Alt, Dir)
                        Xdoc, Ydoc = transform_EN_to_epsg(float(E), float(N), epsg_in, epsg_out_doc)
                        rows.append({"file": os.path.basename(dst), "label": label, "E_in": E, "N_in": N, "lat": lat, "lon": lon,
                                     f"E_{epsg_out_doc}": Xdoc, f"N_{epsg_out_doc}": Ydoc, "hoyde": Alt, "rotasjon": Dir})
                    df_out = pd.DataFrame(rows)
                    st.success(f"Geotagget {len(df_out)} bilder i: {out_dir}")
                    st.download_button("Last ned CSV med posisjoner", df_out.to_csv(index=False).encode("utf-8"),
                                       "geotag_mappe.csv", "text/csv", key="A_csv")

            else:
                if zip_up is None:
                    st.error("Last opp en ZIP med bilder.")
                else:
                    lat, lon = transform_EN_to_wgs84(float(E), float(N), epsg_in)
                    rows = []
                    in_mem = io.BytesIO(zip_up.read())
                    zin = zipfile.ZipFile(in_mem, "r")
                    zout_mem = io.BytesIO()
                    zout = zipfile.ZipFile(zout_mem, "w", zipfile.ZIP_DEFLATED)

                    patt_map = {
                        "Behold originalt navn": "keep",
                        "S_OBJID + originalt navn": "label_orig",
                        "Kun S_OBJID": "label_only",
                        "S_OBJID + avrundet E/N": "label_en",
                    }
                    patt = patt_map[rename_pattern]

                    tempdir = "tmp_zip_in"
                    os.makedirs(tempdir, exist_ok=True)
                    used_names = set()

                    for name in zin.namelist():
                        if not name.lower().endswith((".jpg", ".jpeg")):
                            continue
                        src_bytes = zin.read(name)
                        tmp_path = os.path.join(tempdir, os.path.basename(name))
                        with open(tmp_path, "wb") as f:
                            f.write(src_bytes)

                        write_exif(tmp_path, tmp_path, lat, lon, Alt, Dir)

                        newname = build_new_name(patt, label, os.path.basename(name), E, N)
                        base, ext = os.path.splitext(newname)
                        candidate = newname
                        i = 1
                        while candidate in used_names:
                            candidate = f"{base}_{i}{ext}"
                            i += 1
                        used_names.add(candidate)

                        with open(tmp_path, "rb") as f:
                            zout.writestr(candidate, f.read())

                        Xdoc, Ydoc = transform_EN_to_epsg(float(E), float(N), epsg_in, epsg_out_doc)
                        rows.append({"file": candidate, "label": label, "E_in": E, "N_in": N, "lat": lat, "lon": lon,
                                     f"E_{epsg_out_doc}": Xdoc, f"N_{epsg_out_doc}": Ydoc, "hoyde": Alt, "rotasjon": Dir})
                    zin.close(); zout.close()
                    df_out = pd.DataFrame(rows)
                    st.success(f"Geotagget {len(df_out)} bilder i opplastet ZIP.")
                    st.download_button("Last ned ZIP med geotaggede bilder", data=zout_mem.getvalue(),
                                       file_name="geotagged.zip", mime="application/zip", key="A_zip_dl")
                    st.download_button("Last ned CSV med posisjoner", df_out.to_csv(index=False).encode("utf-8"),
                                       "geotag_zip.csv", "text/csv", key="A_zip_csv")
        except Exception as e:
            st.exception(e)

# ------------- TAB B -------------
with tab_csv:
    st.subheader("B) CSV/Excel-mapping (lokal disk) – Gemini-skjema")
    st.markdown("Format: minst kolonnene **Øst, Nord**. Valgfritt **Høyde, Rotasjon, S_OBJID**. "
                "Velg også hvordan filene skal navngis.")

    colB1, colB2 = st.columns(2)
    with colB1:
        root = st.text_input("Root-mappe (lokal)", value="", key="B_root")
        up = st.file_uploader("Last opp mapping-CSV/Excel", type=["csv","xlsx","xls"], key="B_csv")
        out_root = st.text_input("Ut-root (skriv kopier)", value="", key="B_out")
        overwrite2 = st.checkbox("Overskriv originale filer", value=False, key="B_overwrite")
    with colB2:
        epsg_in2_default = ensure_epsg("Inn-CRS standard (brukes hvis rad mangler epsg)", key_base="B_epsg_in", default=25832)
        epsg_out_doc2 = ensure_epsg("Dokumentasjons-CRS for CSV (kun eksport)", key_base="B_epsg_doc", default=25832)
        rename_pattern_B = st.selectbox("Nytt filnavn (mønster)", [
            "Behold originalt navn",
            "S_OBJID + originalt navn",
            "Kun S_OBJID",
            "S_OBJID + avrundet E/N"
        ], index=1, key="B_rename")

    runB = st.button("Kjør geotag (CSV/Excel)", key="B_run")
    if runB:
        try:
            if not root or not os.path.isdir(root):
                st.error("Root-mappe finnes ikke.")
            elif not up:
                st.error("Last opp en CSV/Excel.")
            else:
                name = up.name.lower()
                if name.endswith((".xlsx",".xls")):
                    dfm = pd.read_excel(up, dtype=str)
                else:
                    dfm = pd.read_csv(up, dtype=str)

                low = {c.lower(): c for c in dfm.columns}
                col_e  = low.get("øst", low.get("oest", low.get("x", None)))
                col_n  = low.get("nord", low.get("y", None))
                col_h  = low.get("høyde", low.get("hoyde", low.get("h", None)))
                col_r  = low.get("rotasjon", low.get("retning", low.get("dir", None)))
                col_id = low.get("s_objid", low.get("navn", low.get("id", None)))
                col_epsg = low.get("epsg", None)

                if not col_e or not col_n:
                    st.error("Fant ikke «Øst»/«Nord» i fila.")
                else:
                    rows = []

                    patt_map = {
                        "Behold originalt navn": "keep",
                        "S_OBJID + originalt navn": "label_orig",
                        "Kun S_OBJID": "label_only",
                        "S_OBJID + avrundet E/N": "label_en",
                    }
                    patt = patt_map[rename_pattern_B]

                    def row_epsg(row):
                        if col_epsg and not pd.isna(row.get(col_epsg, None)):
                            try:
                                return int(row[col_epsg])
                            except Exception:
                                return epsg_in2_default
                        return epsg_in2_default

                    def process_target(path_list, E, N, Alt, Dir, epsg_row, label):
                        lat, lon = transform_EN_to_wgs84(E, N, epsg_row)
                        for p in path_list:
                            if not os.path.isfile(p):
                                st.warning(f"Mangler fil: {p}")
                                continue
                            fname = os.path.relpath(p, root)
                            newname = build_new_name(patt, label, os.path.basename(fname), E, N)
                            if out_root and not overwrite2:
                                dst = os.path.join(out_root, os.path.relpath(os.path.dirname(p), root), newname)
                            else:
                                dst = p if overwrite2 else os.path.join(os.path.dirname(p), newname)
                            os.makedirs(os.path.dirname(dst), exist_ok=True)
                            if dst != p:
                                dst = ensure_unique_path(dst)
                                shutil.copy2(p, dst)
                            write_exif(dst, dst, lat, lon, Alt, Dir)
                            Xdoc, Ydoc = transform_EN_to_epsg(E, N, epsg_row, epsg_out_doc2)
                            rows.append({"file": os.path.relpath(dst, out_root) if out_root else os.path.basename(dst),
                                         "label": label, "E_in": E, "N_in": N, "epsg_in": epsg_row,
                                         "lat": lat, "lon": lon, f"E_{epsg_out_doc2}": Xdoc, f"N_{epsg_out_doc2}": Ydoc,
                                         "hoyde": Alt, "rotasjon": Dir})

                    # Folder-form
                    if {"folder", col_e, col_n}.issubset(dfm.columns):
                        for _, r in dfm.dropna(subset=["folder", col_e, col_n]).iterrows():
                            E   = parse_float_maybe_comma(r[col_e])
                            N   = parse_float_maybe_comma(r[col_n])
                            Alt = parse_float_maybe_comma(r[col_h]) if col_h else None
                            Dir = parse_float_maybe_comma(r[col_r]) if col_r else None
                            label = r[col_id] if col_id else None
                            epsg_row = row_epsg(r)
                            if E is None or N is None:
                                st.warning(f"Hopper over rad med ugyldig E/N: {label or ''}")
                                continue
                            folder = os.path.join(root, str(r["folder"]))
                            jpgs = sorted(glob.glob(os.path.join(folder, "*.jpg")) + glob.glob(os.path.join(folder, "*.JPG")))
                            process_target(jpgs, E, N, Alt, Dir, epsg_row, label)

                    # File-form
                    if {"file", col_e, col_n}.issubset(dfm.columns):
                        for _, r in dfm.dropna(subset=["file", col_e, col_n]).iterrows():
                            E   = parse_float_maybe_comma(r[col_e])
                            N   = parse_float_maybe_comma(r[col_n])
                            Alt = parse_float_maybe_comma(r[col_h]) if col_h else None
                            Dir = parse_float_maybe_comma(r[col_r]) if col_r else None
                            label = r[col_id] if col_id else None
                            epsg_row = row_epsg(r)
                            if E is None or N is None:
                                st.warning(f"Hopper over rad med ugyldig E/N: {label or ''}")
                                continue
                            fpath = os.path.join(root, str(r["file"]))
                            process_target([fpath], E, N, Alt, Dir, epsg_row, label)

                    dfr = pd.DataFrame(rows)
                    st.success(f"Geotagget {len(dfr)} bilder.")
                    st.download_button("Last ned CSV med posisjoner", dfr.to_csv(index=False).encode("utf-8"),
                                       "geotag_csv_mapping.csv", "text/csv", key="B_csv_dl")
        except Exception as e:
            st.exception(e)

st.markdown("---")
st.caption("Filnavn-mønstre: behold originalt, «S_OBJID + originalt», «kun S_OBJID», «S_OBJID + avrundet E/N». "
           "Ugyldige tegn i filnavn erstattes automatisk. Ved navnekonflikt legges _1, _2, ... til.")
