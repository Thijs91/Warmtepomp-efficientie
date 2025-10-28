# ================================================================
# CONFIG
# ================================================================

# InfluxDB verbinding
INFLUX_HOST = "192.168.2.###"   # IP adres van database
INFLUX_PORT = 8086              # Poort van de database
DATABASE = "homeassistant"      # Database naam
USERNAME = "homeassistant"      # Gebruikersnaam
PASSWORD = "wachtwoord"         # Voer hier het wachtwoord in van de database

ROUND_DECIMALS = 3  # aantal decimalen voor afronding

# Delen aan/uit
ENABLE_COP = True  # zet op False om COP/SCOP/stages-deel over te slaan
ENABLE_DDF = True  # zet op False om DDF-deel over te slaan

# JSON en CSV exports
EXPORT_STAGES_JSON    = True                    # Exporteer COP-stages naar JSON-bestanden
EXPORT_CSV            = True                    # Zet op True om CSV's te schrijven
SEP_CSV               = ";"                     # Scheidingsteken voor CSV's
TEMP_TABLE_CSV_PATH   = "temp_bins_cop.csv"     # per-temperatuur-bin export
SCOP_SUMMARY_CSV_PATH = "scop_summary.csv"      # compacte SCOP-export
TEMP_BINS_PNG_PATH    = "temp_bins_cop.png"     # PNG van temp-bins plot
TEMP_POINTS_CSV_PATH  = "temp_bins_points.csv"  # Ruwe punten voor temp-bins plot, geen export ze op None

# Periode: óf LOOKBACK_DAYS, óf TIME_START/END (ISO8601, 'Z')
LOOKBACK_DAYS = 365    # aantal dagen terug vanaf nu; negeer TIME_START/END als deze is ingesteld
TIME_START    = None   # bv. "2025-09-01T00:00:00Z"
TIME_END      = None   # bv. "2025-10-17T23:59:59Z"

#### Metingen & entiteiten 
# (COP/SCOP/stages-blok)
MEAS_COP,  ENT_COP                = "COP",  "cop_verwarmen_wp"                      # COP warmtepomp (Amber) tijdens verwarmen
MEAS_TEMP, ENT_TEMP               = "°C",   "ithodaalderop_ambient_temperature"     # Buiten temperatuur (Amber)
MEAS_TEMP_HUE, ENT_TEMP_HUE       = "°C",   "hue_outdoor_motion_sensor_temperature" # Buiten temperatuur (Hue) voor DDF
MEAS_PWR,  ENT_PWR_F2             = "kW",   "warmtepomp_actueel_vermogen_fase_2"    # Back-up heater (elektra)
MEAS_PWR2, ENT_PWR_F3             = "kW",   "warmtepomp_actueel_vermogen_fase_3"    # Heatpump (elektra)
MEAS_THERMAL_PWR, ENT_THERMAL_PWR = "kW",   "afgegeven_vermogen_wp"                 # Thermisch vermogen (kW_th) voor DDF

# Binaire condities (beiden moeten 'aan' zijn)
MEAS_STATE1, ENT_STATE1 = "state", "ithodaalderop_three_way_valve_state"
MEAS_STATE2, ENT_STATE2 = "state", "ithodaalderop_relais_1"

#### COP/SCOP/stages-blok instellingen
# Resampling / binning
RESAMPLE_RULE = "1min"
TEMP_BIN_WIDTH_C = 0.5
MIN_SAMPLES_PER_TEMPBIN = 5
SHOW_RAW_POINTS_IN_TEMP_PLOT = True

# Stages
K_STAGES     = 8        # Aantal COP-stages
T_MAX_FILTER = 16.0     # None = geen limiet; Maximale buiten temperatuur voor stage-berekening

# Drempelvermogen (kW) om ~0 uit te sluiten (elektrisch)
MIN_VALID_POWER_KW = 0.01

# Plot assen (uniform across figures) (COP/SCOP/stages-blok)
PLOT_XLIM       = (None, None)   # of (0.0, None), (None, 15.0), (0.0, 15.0), of None
PLOT_XTICK_STEP = 2.5            # bv. 5.0; None = geen geforceerde stap (Matplotlib auto)
PLOT_Y1LIM      = (None, None)   # COP-as links
PLOT_Y2LIM      = (0.0, None)    # Vermogen-as rechts (W); min vast, max automatisch

#### DDF (Degree-Day Factor) instellingen
DDF_INDOOR_SETPOINT     = 21.0                  # T_in (°C) voor HDD
DDF_MIN_DAILY_E_KWH     = 0.0                  # drempel voor selectie dagen in DDF (E_kWh ≥ 0 om ~0 uit te sluiten)
DDF_MAX_HOLD_MIN        = 5                     # cap voor power-ffill (min)
DDF_POWER_MIN_THRESHOLD = MIN_VALID_POWER_KW    # kW_th: alleen P > threshold telt mee
DDF_RESAMPLE_MIN        = RESAMPLE_RULE         # gebruikt hetzelfde minuutgrid als hoofdscript 
DDF_LOOKBACK_DAYS       = LOOKBACK_DAYS         # gebruikt hetzelfde venster als hoofdscript
DDF_TIME_START          = TIME_START        
DDF_TIME_END            = TIME_END
DDF_PLOT_BASE           = "ddf"                 # maakt ddf_amber.png en ddf_hue.png

# ================================================================
# IMPORTS
# ================================================================
import numpy as np
import pandas as pd
import datetime as dt
import json
import matplotlib.pyplot as plt

from dateutil import tz
from influxdb import InfluxDBClient
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator, AutoMinorLocator

# ================================================================
# UTILS
# ================================================================
LOCAL_TZ = tz.gettz("Europe/Amsterdam")

def time_window():
    """Bepaal periode o.b.v. config. Retourneert RFC3339 (UTC) zoals 2025-10-21T15:00:00Z."""
    if TIME_START and TIME_END:
        return TIME_START, TIME_END
    t_end = dt.datetime.now(dt.timezone.utc)
    t_start = t_end - dt.timedelta(days=LOOKBACK_DAYS)
    fmt = "%Y-%m-%dT%H:%M:%SZ"  # strakke RFC3339, geen microseconden
    return t_start.strftime(fmt), t_end.strftime(fmt)

def fmt_local(ts):
    if ts is None:
        return "—"
    return ts.tz_convert(LOCAL_TZ).strftime("%Y-%m-%d %H:%M:%S")

def annotate_period_below_axis(ax, begin_ts, end_ts):
    """Plaats 'Begin/Eind' onder x-as."""
    txt = f"Begin: {fmt_local(begin_ts)}\nEind:  {fmt_local(end_ts)}"
    ax.text(0.01, -0.25, txt, transform=ax.transAxes, va="top", ha="left",
            fontsize=9, bbox=dict(boxstyle="round", alpha=0.15, edgecolor="none"))

def set_xlim_from_data(ax, x_array):
    x = np.asarray(x_array, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return
    xmin, xmax = float(np.min(x)), float(np.max(x))
    if xmin == xmax:
        d = 1e-6 if xmin != 0 else 1.0
        ax.set_xlim(xmin - d, xmax + d)
    else:
        ax.set_xlim(xmin, xmax)

def fit_line_with_r2(x_vals, y_vals, n_points=200):
    """Eenvoudige lineaire fit y = a*x + b, geeft (a,b,r2,x_line,y_line) of None."""
    x = np.asarray(x_vals, dtype=float); y = np.asarray(y_vals, dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if x.size < 2 or np.allclose(x, x[0]) or np.allclose(y, y[0]):
        return None
    a, b = np.polyfit(x, y, 1)
    y_hat = a * x + b
    ss_res = np.sum((y - y_hat) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    if ss_tot == 0:
        return None
    r2 = 1 - ss_res / ss_tot
    x_line = np.linspace(np.min(x), np.max(x), n_points)
    y_line = a * x_line + b
    return a, b, r2, x_line, y_line

def safe_merge_asof(left, right):
    if left.empty:
        return right
    if right.empty:
        return left
    return pd.merge_asof(
        left.sort_index(), right.sort_index(),
        left_index=True, right_index=True,
        direction="nearest", tolerance=pd.Timedelta("2min")
    )

def integrate_energy_kwh(ts_index, power_kw):
    """∫ P[kW] dt[h] via rechte-trapjes."""
    if len(ts_index) == 0:
        return 0.0
    t = pd.Series(ts_index)
    dt_hours = (t.shift(-1) - t).dt.total_seconds().div(3600.0).fillna(0.0).to_numpy()
    p = np.asarray(power_kw, dtype=float)
    p[~np.isfinite(p)] = 0.0
    return float(np.sum(p * dt_hours))

# Extra helpers voor DDF
def to_local_index(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    if idx.tz is None:
        return idx.tz_localize("UTC").tz_convert(LOCAL_TZ)
    return idx.tz_convert(LOCAL_TZ)

# ================================================================
# QUERY
# ================================================================
def get_client():
    return InfluxDBClient(
        host=INFLUX_HOST, port=INFLUX_PORT,
        username=USERNAME, password=PASSWORD,
        database=DATABASE
    )

def _query_series(client, measurement, entity_id, t_start, t_end):
    q = f"""
    SELECT time, value
    FROM "{measurement}"
    WHERE time >= '{t_start}' AND time < '{t_end}' AND "entity_id"='{entity_id}'
    ORDER BY time ASC
    """
    result = client.query(q)
    if not result or not list(result.get_points()):
        return pd.DataFrame(columns=["time","value"]).set_index(pd.DatetimeIndex([], name="time"))
    df = pd.DataFrame(list(result.get_points()))
    if df.empty:
        return df
    # robuuste tijd-parsing
    try:
        df["time"] = pd.to_datetime(df["time"], utc=True, format="ISO8601")
    except Exception:
        df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
        if df["time"].isna().any():
            mask = df["time"].isna()
            df.loc[mask, "time"] = pd.to_datetime(
                df.loc[mask, "time"].astype(str).str.replace("Z", "+00:00", regex=False),
                utc=True, errors="coerce"
            )
    df = df.set_index("time").sort_index()
    return df[["value"]].rename(columns={"value": f"{measurement}:{entity_id}"})

def query_series(client, measurement, entity_id, t_start, t_end):
    return _query_series(client, measurement, entity_id, t_start, t_end)

def query_binary_state(client, measurement, entity_id, t_start, t_end, out_name):
    df = _query_series(client, measurement, entity_id, t_start, t_end)
    if df.empty:
        return df
    s = df.iloc[:, 0]
    if s.dtype == "object":
        s_norm = s.astype(str).str.strip().str.lower()
        on_vals  = {"on","true","1","aan","yes","waar"}
        off_vals = {"off","false","0","uit","no","niet waar","onwaar"}
        s_num = np.where(s_norm.isin(on_vals), 1.0,
                 np.where(s_norm.isin(off_vals), 0.0, np.nan))
        s = pd.Series(s_num, index=s.index, name=out_name)
    else:
        s = (pd.to_numeric(s, errors="coerce") > 0).astype(float).rename(out_name)
    return s.to_frame()

# ================================================================
# PREPROCESS
# ================================================================
def resample_mean(df, rule):
    return df.resample(rule).mean()

def resample_state(df, name, rule):
    if df.empty:
        return pd.DataFrame(columns=[name])
    s = df.resample(rule).last().ffill()
    s.columns = [name]
    return s

def build_merged_dataframe(rs_cop, rs_t, rs_f2, rs_f3, rs_st1, rs_st2):
    df = rs_cop
    for part in [rs_t, rs_f2, rs_f3, rs_st1, rs_st2]:
        df = safe_merge_asof(df, part)

    rename_map = {
        f"{MEAS_COP}:{ENT_COP}":      "COP",
        f"{MEAS_TEMP}:{ENT_TEMP}":    "T_outdoor",
        f"{MEAS_PWR}:{ENT_PWR_F2}":   "Power_kW_fase2",
        f"{MEAS_PWR2}:{ENT_PWR_F3}":  "Power_kW_fase3",
    }
    df = df.rename(columns=rename_map)
    if "Power_kW_fase2" not in df.columns: df["Power_kW_fase2"] = 0.0
    if "Power_kW_fase3" not in df.columns: df["Power_kW_fase3"] = 0.0
    df["Power_kW_total"] = df["Power_kW_fase2"].fillna(0) + df["Power_kW_fase3"].fillna(0)
    return df

def apply_filters(df: pd.DataFrame):
    before = len(df)
    df2 = df[(df["ValveOn"] == 1.0) & (df["RelayOn"] == 1.0)]
    after = len(df2)
    df2 = df2.dropna(subset=["COP","Power_kW_total"], how="any")
    return df2, before, after

# ================================================================
# ANALYSE (COP/SCOP/STAGES)
# ================================================================
def _first_last_from_series(s: pd.Series):
    s = s.dropna()
    if s.empty:
        return None, None
    return s.index[0], s.index[-1]

# 1a. Samenvatting per entity + overlap (vóór filters), nu met COP + DDF
def print_entity_summary_and_overlap(
    rs_cop, rs_t, rs_f2, rs_f3, rs_st1, rs_st2,
    rs_temp_hue, rs_therm_pwr
):
    entities = []

    def _collect(group, name, measurement, entity, rs_df):
        if rs_df is None or rs_df.empty:
            entities.append({"group": group, "name": name, "measurement": measurement, "entity": entity, "first": None, "last": None})
            return
        s = rs_df.iloc[:, 0]
        t0, t1 = _first_last_from_series(s)
        entities.append({"group": group, "name": name, "measurement": measurement, "entity": entity, "first": t0, "last": t1})

    # --- COP-blok entiteiten
    _collect("COP", "COP",                          MEAS_COP,   ENT_COP,    rs_cop)
    _collect("COP", "Buitentemp (Amber)",           MEAS_TEMP,  ENT_TEMP,   rs_t)
    _collect("COP", "Vermogen Fase2 (Back-up)",     MEAS_PWR,   ENT_PWR_F2, rs_f2)
    _collect("COP", "Vermogen Fase3 (Heatpump)",    MEAS_PWR2,  ENT_PWR_F3, rs_f3)
    _collect("COP", "Three-way valve (ValveOn)",    MEAS_STATE1, ENT_STATE1, rs_st1)
    _collect("COP", "Relais 1 (RelayOn)",           MEAS_STATE2, ENT_STATE2, rs_st2)

    # --- DDF-blok entiteiten
    _collect("DDF", "Buitentemp (Hue)",             MEAS_TEMP,        ENT_TEMP_HUE,   rs_temp_hue)
    _collect("DDF", "Thermisch vermogen (kW_th)",   MEAS_THERMAL_PWR, ENT_THERMAL_PWR, rs_therm_pwr)

    print("\n=== Registratieperiodes per entity (lokale tijd: Europe/Amsterdam) ===")
    for e in entities:
        meas_ent = f'{e["measurement"]}:{e["entity"]}'
        print(f'- [{e["group"]}] {e["name"]:<32} [{meas_ent}]')
        print(f'    Eerste:  {fmt_local(e["first"])}')
        print(f'    Laatste: {fmt_local(e["last"])}')

    # Compacte lijst (per blok) van gebruikte measurement:entity
    def _compact_list(group_name):
        pairs = [f'{x["measurement"]}:{x["entity"]}' for x in entities if x["group"] == group_name]
        return ", ".join(pairs)

    print("\n— Gebruikte entiteiten —")
    print(f'COP: {_compact_list("COP")}')
    print(f'DDF: {_compact_list("DDF")}')

    # Overlap over ALLE reeksen (handig om snel problemen te zien)
    starts = [e["first"] for e in entities if e["first"] is not None]
    ends   = [e["last"]  for e in entities if e["last"]  is not None]
    if starts and ends:
        overlap_start = max(starts); overlap_end = min(ends); valid = overlap_start <= overlap_end
    else:
        overlap_start = overlap_end = None; valid = False

    print("\n=== Totale overlap (vóór filters; snijvlak van alle reeksen) — lokale tijd ===")
    if valid:
        print(f'Begin: {fmt_local(overlap_start)}')
        print(f'Einde: {fmt_local(overlap_end)}')
    else:
        print("Geen geldige overlap (één of meer reeksen ontbreken of overlappen niet).")

# 1b. Effectieve analyseperiode (na filters)
def print_effective_period_after_filters(df_filtered):
    if df_filtered.empty:
        print("\n=== Effectieve analyseperiode (na filters) — lokale tijd ===")
        print("Dataset leeg na toepassing van filters/vereiste kolommen.")
    else:
        eff_begin, eff_end = df_filtered.index.min(), df_filtered.index.max()
        print("\n=== Effectieve analyseperiode (na filters) — lokale tijd ===")
        print(f'Begin: {fmt_local(eff_begin)}')
        print(f'Einde: {fmt_local(eff_end)}')

# 2. SCOP op gefilterde ruwe data + CSV export
def compute_and_export_scop(df_filtered, t_start, t_end):
    if df_filtered.empty:
        print("\n[Let op] Geen data voor SCOP-berekening na filters.")
        return
    p_el = df_filtered["Power_kW_total"].to_numpy()
    cop  = df_filtered["COP"].to_numpy()
    p_th = cop * p_el  # kW_th

    e_el_kwh = integrate_energy_kwh(df_filtered.index, p_el)
    e_th_kwh = integrate_energy_kwh(df_filtered.index, p_th)
    scop = (e_th_kwh / e_el_kwh) if e_el_kwh > 0 else np.nan

    print("\n=== SCOP (gefilterde ruwe data) ===")
    print(f"Elektrische energie (kWh): {e_el_kwh:,.3f}")
    print(f"Thermische energie (kWh):  {e_th_kwh:,.3f}")
    print(f"SCop:                      {scop:,.3f}")

    if EXPORT_CSV:
        row = [{
            "time_start": t_start,
            "time_end": t_end,
            "samples_after_filters": len(df_filtered),
            "elec_energy_kwh": round(e_el_kwh, ROUND_DECIMALS),
            "thermal_energy_kwh": round(e_th_kwh, ROUND_DECIMALS),
            "SCOP": round(scop, ROUND_DECIMALS),
        }]
        pd.DataFrame(row).to_csv(SCOP_SUMMARY_CSV_PATH, index=False, sep=SEP_CSV)
        print(f"[CSV] SCOP-samenvatting: {SCOP_SUMMARY_CSV_PATH}")

# 3a. Temperatuur-bins + plot + CSV export
def temp_bins_plot_and_export(df_filtered):
    df_temp = df_filtered.dropna(subset=["T_outdoor","COP"]).copy()
    if df_temp.empty:
        print("\n[Waarschuwing] Onvoldoende data (T_outdoor/COP) voor temperatuur-bins.")
        return pd.DataFrame()

    t_min = float(np.floor(df_temp["T_outdoor"].min()))
    t_max = float(np.ceil(df_temp["T_outdoor"].max()))
    if t_min == t_max:
        t_min -= TEMP_BIN_WIDTH_C
        t_max += TEMP_BIN_WIDTH_C

    bins = np.arange(t_min, t_max + TEMP_BIN_WIDTH_C, TEMP_BIN_WIDTH_C)
    df_temp["T_bin"] = pd.cut(df_temp["T_outdoor"], bins=bins, right=False)

    # Zorg dat T_bin_mid gegarandeerd float is (niet category of object)
    df_temp["T_bin_mid"] = df_temp["T_bin"].apply(
        lambda iv: float((iv.left + iv.right) / 2) if pd.notna(iv) else np.nan
    ).astype(float)

    temp_table = (
        df_temp.dropna(subset=["T_bin_mid"])
               .groupby("T_bin_mid", observed=False)
               .agg(
                   COP_mean=("COP","mean"),
                   COP_n=("COP","count"),
                   E_Power_mean=("Power_kW_total","mean"),
                   Power_n=("Power_kW_total","count"),
               )
               .reset_index()
               .sort_values("T_bin_mid")
    )

    temp_table["T_bin_mid"] = pd.to_numeric(temp_table["T_bin_mid"], errors="coerce").astype(float)
    temp_table = temp_table[temp_table["COP_n"] >= MIN_SAMPLES_PER_TEMPBIN]

    if temp_table.empty:
        print("\n[Waarschuwing] Geen temperatuur-bins met genoeg samples voor de plot/export.")
        return pd.DataFrame()

    x = temp_table["T_bin_mid"].values.astype(float)
    y = temp_table["COP_mean"].values.astype(float)

    used_bins = set(temp_table["T_bin_mid"].values)
    df_c = df_temp[df_temp["T_bin_mid"].isin(used_bins)]
    eff_c_begin = df_c.index.min() if not df_c.empty else None
    eff_c_end   = df_c.index.max() if not df_c.empty else None

    plt.figure(figsize=(9,5))
    if SHOW_RAW_POINTS_IN_TEMP_PLOT:
        plt.scatter(
            df_temp["T_outdoor"].values.astype(float),
            df_temp["COP"].values.astype(float),
            alpha=0.15, s=10, label="Ruwe punten (per minuut)"
        )
    fit = fit_line_with_r2(x, y, n_points=200)
    plt.scatter(x, y, s=40, label="Gem. COP per temperatuur-bin")
    if fit is not None:
        a, b, r2, x_line, y_line = fit
        plt.plot(x_line, y_line, label=f"Trend: COP = {a:.3f}·T + {b:.3f} (R²={r2:.3f})")

    plt.xlabel(f"Temperatuur (°C) — bins van {TEMP_BIN_WIDTH_C:g}°C")
    plt.ylabel("Gemiddelde COP")
    plt.title("COP vs Temperatuur (temperatuurbereiken) — gefilterd (ValveOn & RelayOn)")
    ax = plt.gca()
    set_xlim_from_data(ax, x)
    plt.tight_layout(); plt.subplots_adjust(bottom=0.30)
    annotate_period_below_axis(ax, eff_c_begin, eff_c_end)
    plt.legend(); plt.grid(True, alpha=0.3)
    if 'TEMP_BINS_PNG_PATH' in globals() and TEMP_BINS_PNG_PATH:
        fig = plt.gcf()
        fig.savefig(TEMP_BINS_PNG_PATH, dpi=150, bbox_inches="tight")
        print(f"[PNG] Geschreven: {TEMP_BINS_PNG_PATH}")
    plt.show()

    print("\nTemperatuur-bins (middelpunten) met gemiddelde COP, E_Power en n:")
    to_print = temp_table.copy()
    to_print["COP_n"] = to_print["COP_n"].astype(int)
    to_print["Power_n"] = to_print["Power_n"].astype(int)
    print(to_print.round({"T_bin_mid":2, "COP_mean":3, "E_Power_mean":3}).to_string(index=False))

    if EXPORT_CSV:
        temp_table.round(ROUND_DECIMALS).to_csv(TEMP_TABLE_CSV_PATH, index=False, sep=SEP_CSV)
        print(f"[CSV] Geschreven: {TEMP_TABLE_CSV_PATH}")
    
    if EXPORT_CSV and SHOW_RAW_POINTS_IN_TEMP_PLOT and TEMP_POINTS_CSV_PATH:
        pts = (
            df_temp[["T_outdoor", "COP"]]
            .rename(columns={"T_outdoor": "T_outdoor_C", "COP": "COP_raw"})
            .copy()
        )
        # tijdindex als kolom in lokale tijd
        pts = pts.reset_index()
        pts["time_local"] = pts["time"].dt.tz_convert(LOCAL_TZ).dt.strftime("%Y-%m-%d %H:%M:%S")
        pts = pts.drop(columns=["time"])

        # <<< hier afronden: alleen numerieke kolommen >>>
        num_cols = pts.select_dtypes(include=["number"]).columns
        pts[num_cols] = pts[num_cols].round(ROUND_DECIMALS)
        pts.to_csv(TEMP_POINTS_CSV_PATH, index=False, sep=SEP_CSV)
        print(f"[CSV] Ruwe punten (scatter): {TEMP_POINTS_CSV_PATH}")

    return temp_table

# 3b. COP-stages (gewogen segmentatie)
def _pick_power_col(df: pd.DataFrame) -> str:
    for c in ["E_Power_mean", "E_power_mean", "Power_kW_total_mean", "Power_kW_total"]:
        if c in df.columns:
            return c
    raise KeyError("Geen kolom voor gemiddeld elektrisch vermogen gevonden "
                   "(verwacht een van: E_Power_mean, E_power_mean, Power_kW_total_mean, Power_kW_total).")

def compute_cop_stages_from_bins(temp_table: pd.DataFrame,
                                 k: int = K_STAGES,
                                 weight_col: str = "COP_n",
                                 t_col: str = "T_bin_mid",
                                 cop_col: str = "COP_mean",
                                 tmax: float | None = T_MAX_FILTER):
    power_col = _pick_power_col(temp_table)
    req = [t_col, cop_col, weight_col, power_col]
    missing = [c for c in req if c not in temp_table.columns]
    if missing:
        raise KeyError(f"temp_table mist kolommen: {missing}")

    dfb = temp_table.copy()
    dfb[t_col] = pd.to_numeric(dfb[t_col], errors="coerce")
    dfb = dfb.dropna(subset=[t_col, cop_col, weight_col, power_col])

    if tmax is not None:
        dfb = dfb[dfb[t_col] <= tmax]
    dfb = dfb.sort_values(t_col)
    if dfb.empty or dfb[weight_col].sum() <= 0:
        return [], []

    # Gewogen segmentgrenzen (1/k, 2/k, ..., (k-1)/k)
    cumw = dfb[weight_col].cumsum().to_numpy()
    total = cumw[-1]
    targets = np.linspace(1, k - 1, k - 1) / k * total
    cut_idxs = [int(np.searchsorted(cumw, t)) for t in targets]

    # Segmenten slicen
    segments, prev = [], 0
    for idx in cut_idxs:
        segments.append(dfb.iloc[prev:idx]); prev = idx
    segments.append(dfb.iloc[prev:])

    if any(seg.empty for seg in segments):
        segments = list(np.array_split(dfb, k))

    stages, bounds_T = [], []
    for i, seg in enumerate(segments):
        seg = seg.dropna(subset=[cop_col, power_col, weight_col])
        if seg.empty:
            stages.append({"max_power": None, "cop": None})
            continue
        w = seg[weight_col].to_numpy()
        cop_w   = float(np.average(seg[cop_col].to_numpy(),  weights=w))
        pwr_wkw = float(np.average(seg[power_col].to_numpy(), weights=w))
        stages.append({"max_power": int(round(pwr_wkw * 1000)), "cop": round(cop_w, 2)})
        if i < len(segments) - 1:
            bounds_T.append(float(seg[t_col].iloc[-1]))

    return stages, bounds_T

def plot_cop_stages(
    temp_table: pd.DataFrame,
    stages: list,
    bounds_T: list,
    *,
    t_col: str = "T_bin_mid",
    cop_col: str = "COP_mean",
    weight_col: str = "COP_n",
    title: str = "COP-stages uit meetdata",
    save_path: str | None = None,
    # ---- uniforme-as opties (mogen None of partial-None zijn) ----
    xlim: tuple[float, float] | None = PLOT_XLIM,
    xtick_step: float | None = PLOT_XTICK_STEP,
    y1lim: tuple[float, float] | None = PLOT_Y1LIM,   # COP
    y2lim: tuple[float, float] | None = PLOT_Y2LIM,   # Vermogen (W)
):
    """
    Plot COP-stages. As-opties mogen None of (lo, hi) met partial-None zijn.
    Ontbrekende grenzen worden aangevuld met databereik of huidige autoscale.
    """
    def _resolve_limits(req, curr_lo, curr_hi, data_lo=None, data_hi=None):
        """Vul (None, None) of partial-None limieten aan met data- of auto-grenzen."""
        if req is None:
            lo = data_lo if data_lo is not None else curr_lo
            hi = data_hi if data_hi is not None else curr_hi
            return float(lo), float(hi)
        lo, hi = req
        lo = curr_lo if lo is None and data_lo is None else (data_lo if lo is None else lo)
        hi = curr_hi if hi is None and data_hi is None else (data_hi if hi is None else hi)
        return float(lo), float(hi)

    power_col = _pick_power_col(temp_table)

    dfp = temp_table.copy()
    dfp[t_col] = pd.to_numeric(dfp[t_col], errors="coerce")
    dfp = dfp.dropna(subset=[t_col, cop_col, power_col]).sort_values(t_col)

    # --- basisplot
    fig, ax1 = plt.subplots(figsize=(9, 5))

    # Linker as: COP-curve
    ax1.plot(dfp[t_col], dfp[cop_col], label="Gem. COP per bin")
    ax1.set_xlabel("Buitentemperatuur (°C)")
    ax1.set_ylabel("COP_mean")

    # Rechter as: vermogen (W), als stippellijn
    ax2 = ax1.twinx()
    power_W = dfp[power_col].to_numpy(dtype=float) * 1000.0
    ax2.plot(dfp[t_col], power_W, linestyle="--", label="Gem. elektrisch vermogen (W)")
    ax2.set_ylabel("Gemiddeld elektrisch vermogen (W)")

    # Stage-grenzen
    for x in bounds_T:
        ax1.axvline(x, linestyle=":", alpha=0.6, color="gray")

    # Segmenten reconstructie
    all_T = dfp[t_col].to_numpy()
    cut_idx = [int(np.searchsorted(all_T, x, side="right")) for x in bounds_T]
    prev = 0
    segments_idx = []
    for idx in cut_idx:
        segments_idx.append((prev, idx)); prev = idx
    segments_idx.append((prev, len(dfp)))

    for (start, end), st in zip(segments_idx, stages):
        seg = dfp.iloc[start:end]
        if seg.empty or st.get("cop") is None or st.get("max_power") is None:
            continue
        w = seg[weight_col].fillna(1).to_numpy()
        T_mid_w = float(np.average(seg[t_col].to_numpy(), weights=w))
        ax1.scatter(T_mid_w, st["cop"], s=60, edgecolor="k", zorder=6)
        ax2.scatter(T_mid_w, st["max_power"], s=60, edgecolor="k", zorder=6)
        ax1.text(T_mid_w, st["cop"] + 0.1, f"{st['cop']:.2f}",
                 ha="center", va="bottom", fontsize=8, zorder=7)
        ax2.text(T_mid_w, st["max_power"] + 30, f"{int(st['max_power'])}W",
                 ha="center", va="bottom", fontsize=8, zorder=7)

    # UNIFORME AS-INSTELLINGEN
    curr_xlim = ax1.get_xlim()
    curr_y1   = ax1.get_ylim()
    curr_y2   = ax2.get_ylim()

    x_data_lo, x_data_hi   = (float(dfp[t_col].min()), float(dfp[t_col].max())) if not dfp.empty else (curr_xlim[0], curr_xlim[1])
    y1_data_lo, y1_data_hi = (float(dfp[cop_col].min()), float(dfp[cop_col].max())) if not dfp.empty else (curr_y1[0], curr_y1[1])
    y2_data_lo, y2_data_hi = (float(np.nanmin(power_W)) if power_W.size else curr_y2[0],
                              float(np.nanmax(power_W)) if power_W.size else curr_y2[1])

    x_lo, x_hi = _resolve_limits(xlim, curr_xlim[0], curr_xlim[1], x_data_lo, x_data_hi)
    ax1.set_xlim(x_lo, x_hi)

    if xtick_step is not None:
        ax1.xaxis.set_major_locator(MultipleLocator(xtick_step))
        ax1.xaxis.set_minor_locator(AutoMinorLocator(5))

    y1_lo, y1_hi = _resolve_limits(y1lim, curr_y1[0], curr_y1[1], y1_data_lo, y1_data_hi)
    ax1.set_ylim(y1_lo, y1_hi)

    user_hi_is_none = (y2lim is not None and isinstance(y2lim, tuple) and y2lim[1] is None)
    y2_lo, y2_hi = _resolve_limits(y2lim, curr_y2[0], curr_y2[1], y2_data_lo, y2_data_hi)
    if user_hi_is_none:
        y2_hi = float(np.ceil(y2_hi / 100.0) * 100.0)
    ax2.set_ylim(y2_lo, y2_hi)

    # LEGENDA
    stage_proxy = Line2D([0], [0], linestyle=":", color="gray", label="Stagegrens")
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    handles = h1 + h2 + [stage_proxy]
    labels  = l1 + l2 + ["Stagegrens"]
    plt.title(title)
    fig = plt.gcf()
    fig.tight_layout()
    plt.subplots_adjust(bottom=0.40)
    ax1.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.28), ncol=3, framealpha=0.9, borderaxespad=0.0)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()

# ================================================================
# DDF (kWh / K·dag) — geïntegreerde functies
# ================================================================
def q_raw_series(client: InfluxDBClient, measurement: str, entity: str, t_start: str, t_end: str) -> pd.DataFrame:
    q = (
        f'SELECT "value" FROM "{measurement}" '
        f"WHERE time >= '{t_start}' AND time <= '{t_end}' AND \"entity_id\"='{entity}' "
        f"ORDER BY time ASC"
    )
    rs  = client.query(q, database=DATABASE)
    pts = list(rs.get_points()) if rs else []
    if not pts:
        return pd.DataFrame(columns=[entity]).set_index(pd.DatetimeIndex([], name="time"))
    df = pd.DataFrame(pts)
    df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
    df = df.dropna(subset=["time"]).set_index("time").sort_index()
    return df.rename(columns={"value": entity})[[entity]]

def q_last_before(client: InfluxDBClient, measurement: str, entity: str, t_start: str):
    q = (
        f'SELECT last("value") FROM "{measurement}" '
        f"WHERE time < '{t_start}' AND \"entity_id\"='{entity}'"
    )
    rs = client.query(q, database=DATABASE)
    try:
        p = list(rs.get_points())[0]
        return None if p.get("last") is None else str(p.get("last"))
    except Exception:
        return None

def _state_to_bool_series(s: pd.Series) -> pd.Series:
    if s.dtype == "object":
        sn = s.astype(str).str.strip().str.lower()
        truthy = {"on","true","1","1.0","open","active","aan","high","enabled","waar","yes","y","heat","heating"}
        return sn.isin(truthy)
    return pd.to_numeric(s, errors="coerce").fillna(0) > 0

def build_minute_grid_utc(t_start_iso: str, t_end_iso: str) -> pd.DatetimeIndex:
    start = pd.to_datetime(t_start_iso, utc=True)
    end   = pd.to_datetime(t_end_iso,   utc=True)
    return pd.date_range(start=start.floor("min"), end=end.ceil("min"), freq="1min", tz="UTC")

def map_states_to_grid(df_s1: pd.DataFrame, df_s2: pd.DataFrame,
                       grid: pd.DatetimeIndex,
                       seed1, seed2) -> pd.DataFrame:
    def _prep(df_raw: pd.DataFrame, seed):
        if df_raw.empty:
            s = pd.Series(index=grid, data=False)
        else:
            s = df_raw.iloc[:,0]
            if seed is not None and len(grid)>0:
                s = pd.concat([pd.Series([seed], index=[grid[0] - pd.Timedelta(seconds=1)]), s]).sort_index()
            s = _state_to_bool_series(s)
            s = s.reindex(grid, method="ffill").fillna(False)
        return s.astype(bool)

    s1 = _prep(df_s1, seed1)
    s2 = _prep(df_s2, seed2)
    heating_on = (s1 & s2).rename("HEATING_ON")
    out = pd.concat([s1.rename("STATE1"), s2.rename("STATE2"), heating_on], axis=1)
    return out

def map_power_to_grid(df_power: pd.DataFrame, grid: pd.DatetimeIndex,
                      max_hold_min: int | None) -> pd.Series:
    if df_power.empty:
        return pd.Series(index=grid, data=np.nan, name="P_kW")
    p = df_power.iloc[:,0].astype(float)
    if max_hold_min is None:
        p_grid = p.reindex(grid, method="ffill")
    else:
        p_grid = p.reindex(grid, method="ffill", limit=int(max_hold_min))
    return p_grid.rename("P_kW")

def fit_linear(H: np.ndarray, E: np.ndarray):
    n = len(H)
    if n < 3: return None
    x = np.asarray(H, float); y = np.asarray(E, float)
    xm, ym = x.mean(), y.mean()
    Sxx = np.sum((x - xm)**2)
    if Sxx <= 0: return None
    Sxy = np.sum((x - xm)*(y - ym))
    b   = Sxy / Sxx
    a   = ym - b*xm
    yhat= a + b*x
    SSE = np.sum((y - yhat)**2)
    SS  = np.sum((y - ym)**2)
    R2  = 1 - (SSE/SS) if SS>0 else 0.0
    sigma2 = SSE / max(n-2, 1)
    se_b   = np.sqrt(sigma2 / max(Sxx, 1e-12))
    se_a   = np.sqrt(sigma2 * (1.0/n + (xm**2)/max(Sxx, 1e-12)))
    return dict(a=a, b=b, R2=R2, se_a=se_a, se_b=se_b)

def daily_energy_and_hdd(client: InfluxDBClient, t_start: str, t_end: str,
                         grid: pd.DatetimeIndex,
                         P_grid: pd.Series, heating_df: pd.DataFrame,
                         temp_entity: str, indoor_c: float) -> tuple[pd.Series, pd.Series]:
    # Gate op HEATING_ON & drempel
    mask_on = heating_df["HEATING_ON"].reindex(grid).fillna(False)
    P_use   = P_grid.copy()
    P_use[~mask_on] = 0.0
    P_use[(P_use <= DDF_POWER_MIN_THRESHOLD) | (~np.isfinite(P_use))] = 0.0

    # Energie per lokale dag (kWh) — minuutwaarden optellen / 60
    P_local = P_use.copy()
    P_local.index = to_local_index(P_local.index)
    daily_E = (P_local.resample("1D").sum() / 60.0).rename("E_kWh")

    # HDD van gekozen temperatuur-sensor
    dfT = q_raw_series(client, MEAS_TEMP, temp_entity, t_start, t_end)
    if dfT.empty:
        return daily_E, pd.Series(dtype=float, name="HDD")
    sT = dfT[temp_entity].copy()
    sT.index = to_local_index(sT.index)
    T_hourly = sT.resample("1h").mean()
    deg_hours = (indoor_c - pd.to_numeric(T_hourly, errors="coerce")).clip(lower=0)
    daily_HDD = (deg_hours.resample("1D").sum() / 24.0).rename("HDD")
    daily_E   = daily_E.round(ROUND_DECIMALS)
    daily_HDD = daily_HDD.round(ROUND_DECIMALS)
    return daily_E, daily_HDD


def run_ddf_for_sensor(client, t_start, t_end, grid, states_grid, label: str, temp_entity: str,
                       indoor_c: float, plot_base: str):
    # Dag-energie en HDD
    dfP  = q_raw_series(client, MEAS_THERMAL_PWR, ENT_THERMAL_PWR, t_start, t_end)
    if dfP.empty:
        print("[DDF] Power-reeks (thermisch) is leeg binnen venster.")
        return None

    P_grid = map_power_to_grid(dfP, grid, DDF_MAX_HOLD_MIN).fillna(0.0)
    daily_E, daily_HDD = daily_energy_and_hdd(
        client, t_start, t_end, grid, P_grid, states_grid, temp_entity, indoor_c
    )

    if daily_E.empty or daily_HDD.empty:
        print(f"[DDF {label}] Geen overlap HDD/E.")
        return None

    daily = pd.concat([daily_HDD, daily_E], axis=1).dropna()
    daily = daily[(daily["HDD"] > 0) & (daily["E_kWh"] >= float(DDF_MIN_DAILY_E_KWH))]

    if len(daily) < 3:
        print(f"[DDF {label}] Te weinig dagen na filtering (n={len(daily)}; drempel E_kWh≥{DDF_MIN_DAILY_E_KWH}).")
        return None


    corr = np.corrcoef(daily["HDD"], daily["E_kWh"])[0,1]
    print(f"\n[Diag DDF {label}] dagen={len(daily)}  corr(HDD,E)={corr:.3f}  "
          f"mean E={daily['E_kWh'].mean():.2f} kWh/d  mean HDD={daily['HDD'].mean():.2f}")

    fit = fit_linear(daily["HDD"].values, daily["E_kWh"].values)
    if fit is None:
        print(f"[DDF {label}] Fit faalde.")
        return None

    a, b, R2, se_a, se_b = fit["a"], fit["b"], fit["R2"], fit["se_a"], fit["se_b"]
    z = 1.96
    print(f"=== DDF [{label}] (T_in={indoor_c:.1f}°C) ===")
    print(f"Datapunten (dagen): {len(daily)}")
    print(f"Slope (kWh/K·dag):  {b:.{ROUND_DECIMALS}f}  [95% CI {b - z*se_b:.{ROUND_DECIMALS}f} .. {b + z*se_b:.{ROUND_DECIMALS}f}]")
    print(f"Intercept (kWh/d):  {a:.{ROUND_DECIMALS}f}  [95% CI {a - z*se_a:.{ROUND_DECIMALS}f} .. {a + z*se_a:.{ROUND_DECIMALS}f}]")
    print(f"R²:                 {R2:.{ROUND_DECIMALS}f}")

    if plot_base and str(plot_base).lower() != "none":
        H = daily["HDD"].values
        E = daily["E_kWh"].values
        H_grid = np.linspace(0, max(H)*1.05 if len(H) else 1.0, 200)
        E_fit  = a + b*H_grid
        plt.figure(figsize=(7,5))
        plt.scatter(H, E, alpha=0.7, label="Dagen")
        plt.plot(H_grid, E_fit, linewidth=2, label="Lineaire fit")
        plt.xlabel("HDD (K·dag) [lokale dagen]")
        plt.ylabel("Thermische energie (kWh/dag) [lokale dagen]")
        plt.title(
            f"Energy vs HDD [{label}] (T_in={indoor_c:.1f}°C)\n"
            f"Slope={b:.2f}, Intercept={a:.2f}, R²={R2:.2f} | Filter: HDD>0 & E_kWh≥{DDF_MIN_DAILY_E_KWH:g}"
        )
        plt.legend(); plt.tight_layout()
        outp = f"{plot_base}_{label.lower()}.png"
        plt.savefig(outp, dpi=150)
        print(f"Plot opgeslagen: {outp}")
        plt.show()

    if EXPORT_CSV:
        daily_out = daily.copy()
        # lokale datum als kolom
        daily_out = daily_out.reset_index()
        daily_out.rename(columns={"index":"date_local"}, inplace=True)
        # nette string voor datum (zonder tijd)
        if "date_local" in daily_out.columns:
            daily_out["date_local"] = pd.to_datetime(daily_out["date_local"]).dt.strftime("%Y-%m-%d")
        out_csv = f"{plot_base}_{label.lower()}_daily.csv" if plot_base else f"ddf_{label.lower()}_daily.csv"
        daily_out = daily_out.round(ROUND_DECIMALS)
        daily_out.to_csv(out_csv, index=False, sep=SEP_CSV)
        print(f"[CSV] DDF daily dataset ({label}): {out_csv}")

    return dict(label=label, slope=b, intercept=a, R2=R2, n=len(daily))

# ================================================================
# MAIN
# ================================================================
def main():
    # --- Tijdvenster & DB -------------------------------------------------
    t_start, t_end = time_window()
    print(f"Ophalen uit InfluxDB tussen {t_start} en {t_end}")
    client = get_client()

    # --- Reeksen ophalen die we sowieso tonen in Blok 1 -------------------
    df_cop    = query_series(client, MEAS_COP,   ENT_COP,        t_start, t_end)
    df_t      = query_series(client, MEAS_TEMP,  ENT_TEMP,       t_start, t_end)
    df_pwr_f2 = query_series(client, MEAS_PWR,   ENT_PWR_F2,     t_start, t_end)
    df_pwr_f3 = query_series(client, MEAS_PWR2,  ENT_PWR_F3,     t_start, t_end)
    df_st1    = query_binary_state(client, MEAS_STATE1, ENT_STATE1, t_start, t_end, out_name="ValveOn")
    df_st2    = query_binary_state(client, MEAS_STATE2, ENT_STATE2, t_start, t_end, out_name="RelayOn")
    # DDF-gerelateerde entiteiten (voor de lijst in Blok 1)
    df_temp_hue   = query_series(client, MEAS_TEMP,       ENT_TEMP_HUE,    t_start, t_end)
    df_therm_pwr  = query_series(client, MEAS_THERMAL_PWR, ENT_THERMAL_PWR, t_start, t_end)

    # --- Resamplen --------------------------------------------------------
    rs_cop = resample_mean(df_cop, RESAMPLE_RULE)
    rs_t   = resample_mean(df_t,   RESAMPLE_RULE)
    rs_f2  = resample_mean(df_pwr_f2, RESAMPLE_RULE) if not df_pwr_f2.empty else pd.DataFrame(index=rs_cop.index)
    rs_f3  = resample_mean(df_pwr_f3, RESAMPLE_RULE) if not df_pwr_f3.empty else pd.DataFrame(index=rs_cop.index)
    rs_st1 = resample_state(df_st1, "ValveOn", RESAMPLE_RULE)
    rs_st2 = resample_state(df_st2, "RelayOn", RESAMPLE_RULE)

    rs_temp_hue  = resample_mean(df_temp_hue,  RESAMPLE_RULE)
    rs_therm_pwr = resample_mean(df_therm_pwr, RESAMPLE_RULE)

    # --- 1a. Samenvatting + overlap (vóór filters) -----------------------
    print_entity_summary_and_overlap(
        rs_cop, rs_t, rs_f2, rs_f3, rs_st1, rs_st2,
        rs_temp_hue, rs_therm_pwr
    )

    # =========================
    # COP-blok (optioneel)
    # =========================
    if ENABLE_COP:
        # --- Merge + filters ----------------------------------------------
        df = build_merged_dataframe(rs_cop, rs_t, rs_f2, rs_f3, rs_st1, rs_st2)
        df_filtered, before, after = apply_filters(df)
        if before:
            pct = (after / before * 100)
            print(f"\nFilters actief (ValveOn & RelayOn): {after}/{before} rijen over ({pct:.1f}%).")
        else:
            print("\nGeen rijen opgehaald in deze periode.")

        # --- 1b. Effectieve periode na filters ---------------------------
        print_effective_period_after_filters(df_filtered)

        # --- 2. SCOP + export --------------------------------------------
        compute_and_export_scop(df_filtered, t_start, t_end)

        # --- 3. Temperatuur-bins + plot + export -------------------------
        temp_table = temp_bins_plot_and_export(df_filtered)

        if temp_table is not None and not temp_table.empty:
            # ========= Gemeenschappelijke as-instellingen bepalen =========
            def resolve_limits(req, data_lo, data_hi):
                if req is None:
                    return float(data_lo), float(data_hi)
                lo, hi = req
                lo = data_lo if lo is None else lo
                hi = data_hi if hi is None else hi
                return float(lo), float(hi)

            x_data_lo = float(temp_table["T_bin_mid"].min())
            x_data_hi = float(temp_table["T_bin_mid"].max())
            y1_data_lo = float(temp_table["COP_mean"].min())
            y1_data_hi = float(temp_table["COP_mean"].max())

            pcol = "E_Power_mean" if "E_Power_mean" in temp_table.columns else "Power_kW_total"
            power_all_W = temp_table[pcol].to_numpy(dtype=float) * 1000.0
            y2_data_lo = float(np.nanmin(power_all_W)) if power_all_W.size else 0.0
            y2_data_hi = float(np.nanmax(power_all_W)) if power_all_W.size else 1.0

            xlim_common  = resolve_limits(PLOT_XLIM,  x_data_lo,  x_data_hi)
            y1lim_common = resolve_limits(PLOT_Y1LIM, y1_data_lo, y1_data_hi)
            y2lim_common = resolve_limits(PLOT_Y2LIM, y2_data_lo, y2_data_hi)
            if isinstance(PLOT_Y2LIM, tuple) and PLOT_Y2LIM[1] is None:
                y2lim_common = (y2lim_common[0], float(np.ceil(y2lim_common[1] / 100.0) * 100.0))
            xtick_step = PLOT_XTICK_STEP if PLOT_XTICK_STEP is not None else 5.0

            # === Stages berekenen en PLOTTEN met DEZELFDE assen ==========
            stages_all, bounds_all = compute_cop_stages_from_bins(temp_table, k=K_STAGES, tmax=None)
            plot_cop_stages(
                temp_table, stages_all, bounds_all,
                title=f"COP-stages (alle data, k={K_STAGES})",
                save_path=f"cop_stages_Tb_all_k{K_STAGES}.png",
                xlim=xlim_common, xtick_step=xtick_step,
                y1lim=y1lim_common, y2lim=y2lim_common
            )

            if T_MAX_FILTER is not None:
                tmid = pd.to_numeric(temp_table["T_bin_mid"], errors="coerce").astype(float)
                tmp = temp_table.loc[tmid <= float(T_MAX_FILTER)].copy()
                stages_t, bounds_t = compute_cop_stages_from_bins(tmp, k=K_STAGES, tmax=T_MAX_FILTER)
                plot_cop_stages(
                    tmp, stages_t, bounds_t,
                    title=f"COP-stages (T_bin_mid ≤ {T_MAX_FILTER:g} °C, k={K_STAGES})",
                    save_path=f"cop_stages_Tb{int(T_MAX_FILTER)}_k{K_STAGES}.png",
                    xlim=xlim_common, xtick_step=xtick_step,
                    y1lim=y1lim_common, y2lim=y2lim_common
                )

            # --- Export JSON (stages) ------------------------------------
            if EXPORT_STAGES_JSON and 'stages_all' in locals():
                with open("dao_stages_all.json", "w", encoding="utf-8") as f:
                    json.dump({"stages": stages_all}, f, indent=2, ensure_ascii=False)
            if EXPORT_STAGES_JSON and 'stages_t' in locals():
                with open("dao_stages_T.json", "w", encoding="utf-8") as f:
                    json.dump({"stages": stages_t}, f, indent=2, ensure_ascii=False)
        else:
            print("\n[Let op] Geen temp_table, stages worden overgeslagen.")
    else:
        print("\n[INFO] ENABLE_COP = False → COP/SCOP/stages-blok overgeslagen.")

    # =========================
    # DDF-blok (optioneel)
    # =========================
    if ENABLE_DDF:
        print("\n[DDF] Start DDF-analyse…")

        # Seeds voor states en minuutgrid
        seed1 = q_last_before(client, MEAS_STATE1, ENT_STATE1, t_start)
        seed2 = q_last_before(client, MEAS_STATE2, ENT_STATE2, t_start)
        grid  = build_minute_grid_utc(t_start, t_end)

        # States → grid
        dfS1 = q_raw_series(client, MEAS_STATE1, ENT_STATE1, t_start, t_end)
        dfS2 = q_raw_series(client, MEAS_STATE2, ENT_STATE2, t_start, t_end)
        states_grid = map_states_to_grid(dfS1, dfS2, grid, seed1, seed2)

        # Run voor Amber & Hue
        res_amber = run_ddf_for_sensor(client, t_start, t_end, grid, states_grid, "Amber", ENT_TEMP,
                                       DDF_INDOOR_SETPOINT, DDF_PLOT_BASE)
        res_hue   = run_ddf_for_sensor(client, t_start, t_end, grid, states_grid, "Hue", ENT_TEMP_HUE,
                                       DDF_INDOOR_SETPOINT, DDF_PLOT_BASE)

        results = [r for r in [res_amber, res_hue] if r]

        if results:
            print("\n=== Samenvatting DDF ===")
            for r in results:
                print(f"- {r['label']}: slope={r['slope']:.3f} kWh/K·d, intercept={r['intercept']:.3f} kWh/d, R²={r['R2']:.3f}, n={r['n']}")

            # Gemiddelde (eenvoudig rekenkundig) over beschikbare sensoren
            slope_avg     = float(np.mean([r["slope"]     for r in results]))
            intercept_avg = float(np.mean([r["intercept"] for r in results]))
            r2_avg        = float(np.mean([r["R2"]        for r in results]))
            n_sum         = int(np.sum([r["n"]            for r in results]))

            labels_joined = "/".join([r["label"] for r in results])
            print(f"- Gemiddeld ({labels_joined}): slope={slope_avg:.3f} kWh/K·d, "
                  f"intercept={intercept_avg:.3f} kWh/d, R²={r2_avg:.3f}, n={n_sum}")
        else:
            print("\n[Let op] Geen bruikbare DDF-resultaten.")

    else:
        print("\n[INFO] ENABLE_DDF = False → DDF-blok overgeslagen.")

if __name__ == "__main__":
    main()
