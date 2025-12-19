# -*- coding: utf-8 -*-
import sys
import requests
import calendar
from datetime import datetime, timedelta, date
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import json
import hashlib
import time




# ---------- API Config via api.txt ----------

BASE_DIR = Path(__file__).resolve().parent
API_TXT_PATH = BASE_DIR / "../data/api.txt"

_API_CACHE: Optional[dict] = None

def load_api_config() -> dict:
    """
    Reads api.txt with lines:
      RYANAIR_API_URL_TEMPLATE=...
      OPENWEATHER_API_KEY=...
    """
    global _API_CACHE
    if _API_CACHE is not None:
        return _API_CACHE

    if not API_TXT_PATH.exists():
        raise RuntimeError(f"api.txt nicht gefunden: {API_TXT_PATH}")

    cfg = {}
    for raw_line in API_TXT_PATH.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        k, v = line.split("=", 1)
        cfg[k.strip()] = v.strip()

    if "RYANAIR_API_URL_TEMPLATE" not in cfg or not cfg["RYANAIR_API_URL_TEMPLATE"]:
        raise RuntimeError("api.txt: RYANAIR_API_URL_TEMPLATE fehlt oder ist leer.")

    _API_CACHE = cfg
    return cfg

# ---------- Cache ----------

CACHE_ROOT = (BASE_DIR / "../cache").resolve()
RYR_CACHE_DIR = CACHE_ROOT / "ryanair"
OM_CACHE_DIR  = CACHE_ROOT / "openmeteo"

for d in (CACHE_ROOT, RYR_CACHE_DIR, OM_CACHE_DIR):
    d.mkdir(parents=True, exist_ok=True)

CACHE_TTL_SECONDS = 24 * 60 * 60  # 24h

def _cache_dir_for_prefix(prefix: str) -> Path:
    # ryr_* => Ryanair, om_* => Open-Meteo
    if prefix.startswith("ryr_"):
        return RYR_CACHE_DIR
    if prefix.startswith("om_"):
        return OM_CACHE_DIR
    return CACHE_ROOT

def _cache_key(prefix: str, payload: dict) -> Path:
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    h = hashlib.sha256(raw).hexdigest()[:24]
    return _cache_dir_for_prefix(prefix) / f"{prefix}_{h}.json"

def cache_get(prefix: str, payload: dict, ttl_seconds: int = CACHE_TTL_SECONDS) -> Optional[dict]:
    p = _cache_key(prefix, payload)
    if not p.exists():
        return None
    try:
        obj = json.loads(p.read_text(encoding="utf-8"))
        ts = obj.get("_ts", 0)
        if (time.time() - ts) > ttl_seconds:
            return None
        return obj.get("data")
    except Exception:
        return None

def cache_set(prefix: str, payload: dict, data: dict) -> None:
    p = _cache_key(prefix, payload)
    try:
        obj = {"_ts": time.time(), "data": data}
        p.write_text(json.dumps(obj, ensure_ascii=False), encoding="utf-8")
    except Exception:
        pass

def clear_cache(kind: str = "all") -> int:
    """
    kind: 'ryanair' | 'weather' | 'all'
    returns number of deleted files
    """
    if kind == "ryanair":
        dirs = [RYR_CACHE_DIR]
    elif kind == "weather":
        dirs = [OM_CACHE_DIR]
    else:
        dirs = [RYR_CACHE_DIR, OM_CACHE_DIR]

    deleted = 0
    for d in dirs:
        for f in d.glob("*.json"):
            try:
                f.unlink()
                deleted += 1
            except Exception:
                pass
    return deleted

# ---------- HTTP / Daten ----------

CURRENCY_DEFAULT = "EUR"
HEADERS = {"Accept": "application/json", "User-Agent": "RoundtripFinder/2.0 (PySide6+OpenWeather)"}

# IATA -> (lat, lon)
IATA_COORDS: Dict[str, Tuple[float, float]] = {
    "CGN": (50.86590, 7.14270),
    "NRN": (51.6015, 6.1387),
    "PMO": (38.175958, 13.091019),
    "TPS": (37.9114, 12.4880),
}


def fetch_cheapest_per_day_map(origin_iata: str, dest_iata: str, y: int, m: int, curr: str = CURRENCY_DEFAULT) -> Dict[str, dict]:
    cfg = load_api_config()
    url_template = cfg["RYANAIR_API_URL_TEMPLATE"]

    month_str = f"{y:04d}-{m:02d}-01"
    url = url_template.format(origin_iata, dest_iata)
    params = {"outboundMonthOfDate": month_str, "currency": curr}

    cache_payload = {
        "o": origin_iata, "d": dest_iata, "y": y, "m": m, "curr": curr,
        "url": url, "params": params
    }
    cached = cache_get("ryr_month", cache_payload)
    if isinstance(cached, dict):
        return cached

    try:
        r = requests.get(url, params=params, headers=HEADERS, timeout=20)
        r.raise_for_status()
        data = r.json()
    except Exception:
        return {}

    outbound = data.get("outbound", {}) or {}
    fares = outbound.get("fares", []) or []
    result = {}
    for f in fares:
        if f.get("unavailable") or not f.get("price"):
            continue
        day = f.get("day")
        p = f["price"].get("value")
        dep = f.get("departureDate")
        arr = f.get("arrivalDate")
        if day and isinstance(p, (int, float)):
            result[day] = {"price": float(p), "dep": dep, "arr": arr}

    cache_set("ryr_month", cache_payload, result)
    return result



def hhmm(ts: Optional[str]) -> str:
    if not ts:
        return "-"
    try:
        return ts.split("T")[1][:5]
    except Exception:
        return "-"


def fetch_temp_last_year(lat: float, lon: float, d: date) -> Optional[float]:
    try:
        ly = d.replace(year=d.year - 1)
    except ValueError:
        ly = d.replace(year=d.year - 1, day=28)

    cache_payload = {"lat": lat, "lon": lon, "date": ly.isoformat(), "metric": "tmax"}
    cached = cache_get("om_day", cache_payload)
    if isinstance(cached, dict) and "t" in cached:
        return None if cached["t"] is None else float(cached["t"])

    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": ly.isoformat(),
        "end_date": ly.isoformat(),
        "daily": "temperature_2m_max",
        "timezone": "UTC",
    }

    try:
        r = requests.get(url, params=params, timeout=20)
        r.raise_for_status()
        j = r.json()

        daily = j.get("daily")
        if not daily:
            cache_set("om_day", cache_payload, {"t": None})
            return None

        temps = daily.get("temperature_2m_max")
        if not temps:
            cache_set("om_day", cache_payload, {"t": None})
            return None

        t = temps[0]
        out = float(t) if isinstance(t, (int, float)) else None
        cache_set("om_day", cache_payload, {"t": out})
        return out
    except Exception:
        return None



# ---------- Model ----------

@dataclass
class DayQuote:
    day: str
    price: float
    depISO: Optional[str]
    arrISO: Optional[str]


@dataclass
class Candidate:
    total: float
    out_day: str
    ret_day: str
    out_price: float
    ret_price: float
    dep_o: Optional[str]
    arr_o: Optional[str]
    dep_r: Optional[str]
    arr_r: Optional[str]
    route_label: str
    origin_iata: str
    dest_iata: str

    out_temp_ly: Optional[float] = None
    ret_temp_ly: Optional[float] = None
    score: Optional[float] = None



# ---------- Roundtrip-Suche ----------

def months_between(start: date, end: date) -> List[Tuple[int, int]]:
    ym = []
    y, m = start.year, start.month
    while True:
        ym.append((y, m))
        if y == end.year and m == end.month:
            break
        m += 1
        if m == 13:
            m = 1
            y += 1
    return ym


def find_roundtrips_for_route_by_dates(
    origin: str,
    dest: str,
    start_date: date,
    end_date: date,
    min_days: int,
    max_days: int,
    currency: str,
    month_cache: Dict[Tuple[str, str, int, int, str], Dict[str, DayQuote]],
) -> List[Candidate]:

    def get_month_map(o: str, d: str, y: int, m: int, c: str) -> Dict[str, DayQuote]:
        key = (o, d, y, m, c)
        if key in month_cache:
            return month_cache[key]
        raw = fetch_cheapest_per_day_map(o, d, y, m, c)
        mapped = {k: DayQuote(k, v["price"], v.get("dep"), v.get("arr")) for k, v in raw.items()}
        month_cache[key] = mapped
        return mapped

    ym_list = months_between(start_date, end_date)
    out_maps: Dict[Tuple[int, int], Dict[str, DayQuote]] = {}
    ret_maps: Dict[Tuple[int, int], Dict[str, DayQuote]] = {}

    for (y, m) in ym_list:
        out_maps[(y, m)] = get_month_map(origin, dest, y, m, currency)
    for (y, m) in ym_list:
        ret_maps[(y, m)] = get_month_map(dest, origin, y, m, currency)

    cands: List[Candidate] = []
    for fmap in out_maps.values():
        for out_day, info in fmap.items():
            try:
                out_dt = datetime.strptime(out_day, "%Y-%m-%d").date()
            except ValueError:
                continue
            if not (start_date <= out_dt <= end_date):
                continue

            for span in range(min_days, max_days + 1):
                ret_dt = out_dt + timedelta(days=span)
                if not (start_date <= ret_dt <= end_date):
                    continue

                ret_map = ret_maps.get((ret_dt.year, ret_dt.month), {})
                ret_day = ret_dt.strftime("%Y-%m-%d")
                if ret_day not in ret_map:
                    continue

                out_price = info.price
                ret_price = ret_map[ret_day].price
                total = out_price + ret_price

                cands.append(
                    Candidate(
                        total=total,
                        out_day=out_day,
                        ret_day=ret_day,
                        out_price=out_price,
                        ret_price=ret_price,
                        dep_o=info.depISO,
                        arr_o=info.arrISO,
                        dep_r=ret_map[ret_day].depISO,
                        arr_r=ret_map[ret_day].arrISO,
                        route_label=f"{origin} ↔ {dest}",
                        origin_iata=origin,
                        dest_iata=dest,
                    )
                )

    cands.sort(key=lambda x: (x.total, x.out_day))
    return cands


# ---------- GUI ----------

from PySide6.QtCore import Qt, QThread, Signal, QObject, QDate
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QGridLayout, QHBoxLayout, QVBoxLayout,
    QLabel, QComboBox, QLineEdit, QSpinBox, QPushButton,
    QGroupBox, QTableWidget, QTableWidgetItem, QSplitter, QMessageBox,
    QTabWidget, QScrollArea, QDateEdit, QFileDialog, QSlider, QCheckBox
)

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.dates as mdates


def fmt_temp(t: Optional[float]) -> str:
    return "keine Temperaturdaten" if t is None else f"{t:.1f}°C"

def parse_hhmm_from_iso(ts: Optional[str]) -> Optional[Tuple[int, int]]:
    # "2025-12-19T06:40:00.000" -> (6,40)
    if not ts or "T" not in ts:
        return None
    try:
        hhmm = ts.split("T")[1][:5]
        h, m = hhmm.split(":")
        return int(h), int(m)
    except Exception:
        return None

def time_in_range(ts: Optional[str], min_h: int, max_h: int) -> bool:
    hm = parse_hhmm_from_iso(ts)
    if hm is None:
        return False
    h, _ = hm
    return (min_h <= h <= max_h)


class PriceChart(FigureCanvas):
    def __init__(self, parent=None):
        fig = Figure(figsize=(7, 4), tight_layout=True)
        self.ax = fig.add_subplot(111)
        super().__init__(fig)
        self.setParent(parent)

    def plot_routes(self, price_series: Dict[str, List[Tuple[date, float]]], color_map: Optional[Dict[str, str]] = None) -> Dict[str, str]:
        self.ax.clear()

        if color_map is None:
            color_map = {}

        for label, data in price_series.items():
            if not data:
                continue
            data_sorted = sorted(data, key=lambda t: t[0])
            xs = [d for d, _ in data_sorted]
            ys = [p for _, p in data_sorted]

            line, = self.ax.plot(xs, ys, linestyle="solid", label=label)
            # Farbe merken (wenn noch nicht gesetzt)
            if label not in color_map:
                color_map[label] = line.get_color()

        self.ax.set_title("Beste Gesamtpreise pro Abflugtag (Routenvergleich)")
        self.ax.set_xlabel("Abflugdatum")
        self.ax.set_ylabel("Gesamtpreis")
        self.ax.grid(True, which="both", alpha=0.3)
        self.ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
        self.ax.tick_params(axis="x", rotation=45)

        handles, labels = self.ax.get_legend_handles_labels()
        if labels:
            self.ax.legend()
        else:
            self.ax.text(0.5, 0.5, "Keine Daten für diesen Zeitraum",
                        transform=self.ax.transAxes, ha="center", va="center", alpha=0.6)

        self.draw()
        return color_map

class TempHeatmapChart(FigureCanvas):
    def __init__(self, parent=None):
        fig = Figure(figsize=(7, 4), constrained_layout=True)
        self.ax = fig.add_subplot(111)
        super().__init__(fig)
        self.setParent(parent)
        self._cbar = None  # <--- merken

    def plot_heatmap(self, temp_heatmap: Dict[str, Dict[date, float]], color_map=None):
        self.ax.clear()

        # <--- alte Colorbar entfernen, wenn vorhanden
        if self._cbar is not None:
            try:
                self._cbar.remove()
            except Exception:
                try:
                    self._cbar.ax.remove()
                except Exception:
                    pass
            self._cbar = None

        all_dates = sorted({d for mp in temp_heatmap.values() for d in mp.keys()})
        labels = list(temp_heatmap.keys())

        if not all_dates or not labels:
            self.ax.text(0.5, 0.5, "Keine Temperaturdaten verfügbar",
                         transform=self.ax.transAxes, ha="center", va="center", alpha=0.6)
            self.draw()
            return

        import numpy as np
        mat = np.full((len(labels), len(all_dates)), np.nan, dtype=float)
        for i, lab in enumerate(labels):
            mp = temp_heatmap.get(lab, {})
            for j, d in enumerate(all_dates):
                if d in mp:
                    mat[i, j] = mp[d]

        im = self.ax.imshow(mat, aspect="auto", interpolation="nearest", cmap="coolwarm")
        im.set_clim(vmin=0, vmax=35)

        self.ax.set_title("Temperatur-Heatmap (letztes Jahr) am Ziel – pro Abflugdatum")
        self.ax.set_yticks(range(len(labels)))
        self.ax.set_yticklabels(labels)

        step = max(1, len(all_dates) // 10)
        xt = list(range(0, len(all_dates), step))
        self.ax.set_xticks(xt)
        self.ax.set_xticklabels([all_dates[k].strftime("%Y-%m-%d") for k in xt], rotation=45, ha="right")

        # <--- Colorbar speichern
        self._cbar = self.figure.colorbar(im, ax=self.ax)
        self._cbar.set_label("°C")

        self.draw()


class Worker(QObject):
    finished = Signal(dict, list, dict, dict, dict, str)
    # per_route, combined, price_series, temp_series_line, temp_heatmap, error


    def __init__(self, params):
        super().__init__()
        self.params = params

    def run(self):
        p = self.params
        currency = p["currency"]
        min_days = p["min_days"]
        max_days = p["max_days"]
        routes = p["routes"]
        top_n = p["top_n"]

        # Temp-Filter (optional)
        use_weather_filter = p.get("use_weather_filter", False)
        require_weather = p.get("require_weather", False)
        ideal_temp = float(p.get("ideal_temp", 23.0))
        temp_tol   = float(p.get("temp_tol", 5.0))

        # Zeitfilter (Outbound)
        use_time = p.get("use_time", False)
        out_dep_min_h = p.get("out_dep_min_h", 0)
        out_dep_max_h = p.get("out_dep_max_h", 23)
        ret_dep_min_h = p.get("ret_dep_min_h", 0)
        ret_dep_max_h = p.get("ret_dep_max_h", 23)


        month_cache: Dict[Tuple[str, str, int, int, str], Dict[str, DayQuote]] = {}
        per_route: Dict[str, List[Candidate]] = {}
        all_cands: List[Candidate] = []

        price_series: Dict[str, List[Tuple[date, float]]] = {}
        temp_series: Dict[str, List[Tuple[date, float]]] = {}  # Linie (wie vorher)

        # Heatmap: label -> dict(date -> temp)
        temp_heatmap: Dict[str, Dict[date, float]] = {}


        weather_cache: Dict[Tuple[float, float, str], Optional[float]] = {}

        def get_temp_ly_for_iata(iata: str, day_str: str) -> Optional[float]:
            coords = IATA_COORDS.get(iata)
            if not coords:
                return None

            lat, lon = coords
            key = (lat, lon, day_str)
            if key in weather_cache:
                return weather_cache[key]

            try:
                d0 = datetime.strptime(day_str, "%Y-%m-%d").date()
            except ValueError:
                weather_cache[key] = None
                return None

            t = fetch_temp_last_year(lat, lon, d0)
            weather_cache[key] = t
            return t

        error = ""
        combined: List[Candidate] = []
        if len(routes) >= 2 and all_cands:
            all_cands.sort(key=lambda x: (x.score if x.score is not None else 1e18, x.total, x.out_day))
            combined = all_cands[:top_n]


        try:
            start_date: date = p["start_date"]
            end_date: date = p["end_date"]

            for (o, d) in routes:
                label = f"{o} ↔ {d}"
                cands = find_roundtrips_for_route_by_dates(
                    o, d, start_date, end_date, min_days, max_days, currency, month_cache
                )

                # Wetter & Score + Filter
                filtered: List[Candidate] = []
                for c in cands:
                    # Zeitfilter: Abflugstunden müssen in Range liegen
                    use_time = p.get("use_time", False)

                    # Zeitfilter nur wenn aktiv
                    if use_time:
                        if not time_in_range(c.dep_o, out_dep_min_h, out_dep_max_h):
                            continue
                        if not time_in_range(c.dep_r, ret_dep_min_h, ret_dep_max_h):
                            continue

                    # Temperaturen IMMER versuchen zu holen
                    c.out_temp_ly = get_temp_ly_for_iata(c.dest_iata, c.out_day)
                    c.ret_temp_ly = get_temp_ly_for_iata(c.dest_iata, c.ret_day)

                    # Wetterfilter nur wenn aktiv
                    if use_weather_filter:
                        temps = [t for t in (c.out_temp_ly, c.ret_temp_ly) if t is not None]

                        # wenn Filter an und "require": ohne Temps raus
                        if require_weather and not temps:
                            continue

                        # wenn Temps da sind: prüfen
                        if temps:
                            avg_t = sum(temps) / len(temps)
                            if not (ideal_temp - temp_tol <= avg_t <= ideal_temp + temp_tol):
                                continue

                    c.score = c.total
                    filtered.append(c)

                filtered.sort(key=lambda x: (x.score if x.score is not None else 1e18, x.total, x.out_day))

                rows = filtered[:top_n]
                per_route[label] = rows


                best_price_per_day: Dict[str, float] = {}
                best_temp_per_day: Dict[date, float] = {}

                # Für Diagramme: pro Out-Day den besten Preis (aus gefilterten Kandidaten)
                for c in filtered:
                    best_price_per_day[c.out_day] = min(best_price_per_day.get(c.out_day, float("inf")), c.total)

                    # Heatmap-Temp: nutze out_temp_ly am Ziel
                    if c.out_temp_ly is not None:
                        try:
                            od = datetime.strptime(c.out_day, "%Y-%m-%d").date()
                            # wenn mehrfach: nimm z.B. max oder mean; wir nehmen "max" hier
                            best_temp_per_day[od] = max(best_temp_per_day.get(od, -1e9), c.out_temp_ly)
                        except ValueError:
                            pass

                p_points: List[Tuple[date, float]] = []
                t_points: List[Tuple[date, float]] = []

                for day_str, best_total in best_price_per_day.items():
                    try:
                        dt_day = datetime.strptime(day_str, "%Y-%m-%d").date()
                    except ValueError:
                        continue
                    p_points.append((dt_day, best_total))

                # Linie: aus best_temp_per_day
                for dday, t in best_temp_per_day.items():
                    t_points.append((dday, t))

                price_series[label] = sorted(p_points, key=lambda x: x[0])
                temp_series[label] = sorted(t_points, key=lambda x: x[0])
                temp_heatmap[label] = best_temp_per_day

            combined_pool: List[Candidate] = []
            for rows in per_route.values():
                combined_pool.extend(rows)

            combined_pool.sort(key=lambda x: (x.total, x.out_day))
            combined = combined_pool[:top_n]

            if len(routes) >= 2 and all_cands:
                all_cands.sort(key=lambda x: (x.total, x.out_day))
                combined = all_cands[:top_n]
                for c in combined:
                    c.out_temp_ly = get_temp_ly_for_iata(c.dest_iata, c.out_day)
                    c.ret_temp_ly = get_temp_ly_for_iata(c.dest_iata, c.ret_day)

        except Exception as e:
            error = str(e)

        no_prices = all((not pts) for pts in price_series.values())
        # Preise fehlen => nur dann wirklich "fatal"
        if no_prices:
            error = "Keine Flüge/Preise in diesem Zeitraum gefunden (Ryanair API hat keine Daten geliefert)."

        self.finished.emit(per_route, combined, price_series, temp_series, temp_heatmap, error)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.last_per_route = {}
        self.last_combined = []
        self.last_price_series = {}
        self.last_temp_series = {}
        self.last_temp_heatmap = {}
        self.view_temp_heatmap = {}
        self.view_per_route = {}
        self.view_combined = []
        self.view_price_series = {}
        self.view_temp_series = {}
        self.last_color_map = {}
        self.base_start_date = None
        self.base_end_date = None

        self.setWindowTitle("Roundtrip-Finder (Ryanair + OpenWeather)")
        self.resize(1280, 780)

        self.mode = QComboBox()
        self.mode.addItems(["A: CGN↔PMO & NRN↔TPS", "B: 2 eigene Routen", "C: 1 Route"])

        self.o1 = QLineEdit("CGN"); self.d1 = QLineEdit("PMO")
        self.o2 = QLineEdit("NRN"); self.d2 = QLineEdit("TPS")
        for w in (self.o1, self.d1, self.o2, self.d2):
            w.setMaxLength(3); w.setPlaceholderText("IATA"); w.setInputMask(">AAA;_")

        today = date.today()
        self.dStart = QDateEdit(); self.dStart.setDisplayFormat("yyyy-MM-dd")
        self.dStart.setCalendarPopup(True)
        self.dStart.setDate(QDate(today.year, today.month, today.day))

        self.dEnd = QDateEdit(); self.dEnd.setDisplayFormat("yyyy-MM-dd")
        self.dEnd.setCalendarPopup(True)
        default_end = today + timedelta(days=30)
        self.dEnd.setDate(QDate(default_end.year, default_end.month, default_end.day))

        self.minDays = QSpinBox(); self.minDays.setRange(1, 90); self.minDays.setValue(3)
        self.maxDays = QSpinBox(); self.maxDays.setRange(1, 360); self.maxDays.setValue(14)
        self.topN = QSpinBox(); self.topN.setRange(1, 50); self.topN.setValue(5)

        self.currency = QComboBox(); self.currency.addItems(["EUR", "GBP", "PLN", "USD"])

        self.btnSearch = QPushButton("Suchen")
        self.btnSearch.clicked.connect(self.on_search)

        self.tabs = QTabWidget()

        self.results_container = QWidget()
        self.results_layout = QVBoxLayout(self.results_container)
        self.results_layout.setContentsMargins(8, 8, 8, 8)
        self.results_layout.setSpacing(8)

        self.results_scroll = QScrollArea()
        self.results_scroll.setWidgetResizable(True)
        self.results_scroll.setWidget(self.results_container)
        self.tabs.addTab(self.results_scroll, "Ergebnisse")

        # Tab 2: Preise
        self.priceChart = PriceChart()
        price_wrap = QWidget()
        price_lay = QVBoxLayout(price_wrap)
        price_lay.setContentsMargins(8, 8, 8, 8)
        price_lay.addWidget(self.priceChart)
        self.tabs.addTab(price_wrap, "Preise")

        # Tab 3: Wetter
        self.tempChart = TempHeatmapChart()
        temp_wrap = QWidget()
        temp_lay = QVBoxLayout(temp_wrap)
        temp_lay.setContentsMargins(8, 8, 8, 8)
        temp_lay.addWidget(self.tempChart)
        self.tabs.addTab(temp_wrap, "Wetter")


        left = QVBoxLayout()

        g_mode = QGroupBox("Modus")
        lm = QVBoxLayout(g_mode)
        lm.addWidget(self.mode)
        left.addWidget(g_mode)

        g_routes = QGroupBox("Routen")
        grid = QGridLayout(g_routes)
        grid.addWidget(QLabel("Route 1:"), 0, 0)
        grid.addWidget(self.o1, 0, 1); grid.addWidget(QLabel("→"), 0, 2); grid.addWidget(self.d1, 0, 3)
        grid.addWidget(QLabel("Route 2:"), 1, 0)
        grid.addWidget(self.o2, 1, 1); grid.addWidget(QLabel("→"), 1, 2); grid.addWidget(self.d2, 1, 3)
        left.addWidget(g_routes)

        # =========================
        # A) Datum
        # =========================
        g_date = QGroupBox("Nach Datum filtern")
        gd = QGridLayout(g_date)

        gd.addWidget(QLabel("Start-Tag"), 0, 0); gd.addWidget(self.dStart, 0, 1)
        gd.addWidget(QLabel("End-Tag"),   0, 2); gd.addWidget(self.dEnd,   0, 3)
        gd.addWidget(QLabel("Min. Tage"), 1, 0); gd.addWidget(self.minDays, 1, 1)
        gd.addWidget(QLabel("Max. Tage"), 1, 2); gd.addWidget(self.maxDays, 1, 3)

        left.addWidget(g_date)


        # =========================
        # B) Uhrzeit
        # =========================
        g_time = QGroupBox("Nach Uhrzeit filtern")
        gt = QGridLayout(g_time)

        self.cbUseTime = QCheckBox("Uhrzeit-Filter aktivieren")
        self.cbUseTime.setChecked(False)

        self.spOutDepMin = QSpinBox(); self.spOutDepMin.setRange(0, 23)
        self.spOutDepMax = QSpinBox(); self.spOutDepMax.setRange(0, 23)
        self.spRetDepMin = QSpinBox(); self.spRetDepMin.setRange(0, 23)
        self.spRetDepMax = QSpinBox(); self.spRetDepMax.setRange(0, 23)

        gt.addWidget(self.cbUseTime, 0, 0, 1, 4)
        gt.addWidget(QLabel("Hinflug Abflug (min/max)"), 1, 0)
        gt.addWidget(self.spOutDepMin, 1, 1); gt.addWidget(self.spOutDepMax, 1, 2)
        gt.addWidget(QLabel("Rückflug Abflug (min/max)"), 2, 0)
        gt.addWidget(self.spRetDepMin, 2, 1); gt.addWidget(self.spRetDepMax, 2, 2)

        left.addWidget(g_time)


        # =========================
        # C) Wetter
        # =========================
        g_weather = QGroupBox("Wetter (Temperaturen anzeigen / optional filtern)")
        gw = QGridLayout(g_weather)

        # 1) Wetter-Filter AUS per Default
        self.cbUseWeatherFilter = QCheckBox("Nach Wetter filtern")
        self.cbUseWeatherFilter.setChecked(False)

        # Optional: wenn Filter an ist, aber keine Temperaturen => Ergebnis verwerfen
        self.cbRequireWeather = QCheckBox("Beim Filtern: nur Ergebnisse mit Temperaturdaten")
        self.cbRequireWeather.setChecked(False)

        # 2) Ideal + Toleranz
        self.spIdealTemp = QSpinBox(); self.spIdealTemp.setRange(-10, 40); self.spIdealTemp.setValue(23)
        self.spTempTol   = QSpinBox(); self.spTempTol.setRange(0, 20); self.spTempTol.setValue(5)

        gw.addWidget(self.cbUseWeatherFilter, 0, 0, 1, 4)
        gw.addWidget(self.cbRequireWeather,   1, 0, 1, 4)

        gw.addWidget(QLabel("Idealtemperatur (°C)"), 2, 0); gw.addWidget(self.spIdealTemp, 2, 1)
        gw.addWidget(QLabel("± Toleranz (°C)"),      2, 2); gw.addWidget(self.spTempTol,   2, 3)

        left.addWidget(g_weather)

        # Toggle-Logik
        self.cbUseWeatherFilter.stateChanged.connect(self.update_weather_ui_state)
        self.cbRequireWeather.stateChanged.connect(self.update_weather_ui_state)
        self.update_weather_ui_state()

        left.addWidget(self.btnSearch)
        left.addStretch(1)

        self.btnExportCSV = QPushButton("Export CSV")
        self.btnExportPNG = QPushButton("Export PNG")

        self.btnExportCSV.clicked.connect(self.export_csv)
        self.btnExportPNG.clicked.connect(self.export_png)

        self.btnExportCSV.setEnabled(False)
        self.btnExportPNG.setEnabled(False)

        left.addWidget(self.btnExportCSV)
        left.addWidget(self.btnExportPNG)


        g_filter = QGroupBox("Datum-Filter (nach Suche)")
        gf = QGridLayout(g_filter)

        self.sStart = QSlider()
        self.sStart.setOrientation(Qt.Horizontal)
        self.sEnd = QSlider()
        self.sEnd.setOrientation(Qt.Horizontal)

        self.lblFStart = QLabel("Start: -")
        self.lblFEnd = QLabel("Ende: -")

        self.sStart.valueChanged.connect(self.on_filter_changed)
        self.sEnd.valueChanged.connect(self.on_filter_changed)

        self.sStart.setEnabled(False)
        self.sEnd.setEnabled(False)

        gf.addWidget(self.lblFStart, 0, 0); gf.addWidget(self.sStart, 0, 1)
        gf.addWidget(self.lblFEnd,   1, 0); gf.addWidget(self.sEnd,   1, 1)

        left.addWidget(g_filter)

        g_cache = QGroupBox("Cache")
        gc = QGridLayout(g_cache)

        self.btnClearRyr = QPushButton("Cache leeren (Flüge)")
        self.btnClearWx  = QPushButton("Cache leeren (Wetter)")
        self.btnClearAll = QPushButton("Cache leeren (Alles)")

        self.btnClearRyr.clicked.connect(self.on_clear_cache_ryr)
        self.btnClearWx.clicked.connect(self.on_clear_cache_weather)
        self.btnClearAll.clicked.connect(self.on_clear_cache_all)

        gc.addWidget(self.btnClearRyr, 0, 0)
        gc.addWidget(self.btnClearWx,  1, 0)
        gc.addWidget(self.btnClearAll, 2, 0)

        left.addWidget(g_cache)


        splitter = QSplitter()
        left_widget = QWidget(); left_widget.setLayout(left)
        splitter.addWidget(left_widget)
        splitter.addWidget(self.tabs)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)

        central = QWidget()
        lay = QHBoxLayout(central)
        lay.addWidget(splitter)
        self.setCentralWidget(central)
        self.statusBar().showMessage("Bereit.")


        self.mode.currentIndexChanged.connect(self.update_mode_fields)
        self.update_mode_fields()

    def update_mode_fields(self):
        idx = self.mode.currentIndex()
        if idx == 0:
            self.o1.setText("CGN"); self.d1.setText("PMO")
            self.o2.setText("NRN"); self.d2.setText("TPS")
            self.o1.setEnabled(False); self.d1.setEnabled(False)
            self.o2.setEnabled(False); self.d2.setEnabled(False)
        elif idx == 1:
            self.o1.setEnabled(True); self.d1.setEnabled(True)
            self.o2.setEnabled(True); self.d2.setEnabled(True)
        else:
            self.o1.setEnabled(True); self.d1.setEnabled(True)
            self.o2.setEnabled(False); self.d2.setEnabled(False)

    def update_weather_ui_state(self):
        use_filter = self.cbUseWeatherFilter.isChecked()
        self.cbRequireWeather.setEnabled(use_filter)
        self.spIdealTemp.setEnabled(use_filter)
        self.spTempTol.setEnabled(use_filter)

        if not use_filter and self.cbRequireWeather.isChecked():
            self.cbRequireWeather.setChecked(False)

    def clear_tables(self):
        while self.results_layout.count():
            item = self.results_layout.takeAt(0)
            w = item.widget()
            if w:
                w.deleteLater()

    def on_filter_changed(self):
        if not self.base_start_date or not self.base_end_date:
            return

        a = self.sStart.value()
        b = self.sEnd.value()

        # sicherstellen start <= end
        if a > b:
            # einfach den anderen Slider nachziehen
            sender = self.sender()
            if sender is self.sStart:
                self.sEnd.setValue(a)
                b = a
            else:
                self.sStart.setValue(b)
                a = b

        fstart = self.base_start_date + timedelta(days=a)
        fend = self.base_start_date + timedelta(days=b)

        self.lblFStart.setText(f"Start: {fstart.isoformat()}")
        self.lblFEnd.setText(f"Ende: {fend.isoformat()}")

        self.apply_filter(fstart, fend)

    def apply_filter(self, fstart: date, fend: date):
        # Kandidaten filtern (nach out_day)
        def in_range(c: Candidate) -> bool:
            try:
                od = datetime.strptime(c.out_day, "%Y-%m-%d").date()
                return fstart <= od <= fend
            except ValueError:
                return False

        self.view_per_route = {k: [c for c in rows if in_range(c)] for k, rows in self.last_per_route.items()}
        self.view_combined = [c for c in self.last_combined if in_range(c)]

        # Serien filtern
        def filter_series(series: Dict[str, List[Tuple[date, float]]]) -> Dict[str, List[Tuple[date, float]]]:
            out = {}
            for label, pts in series.items():
                out[label] = [(d, v) for (d, v) in pts if fstart <= d <= fend]
            return out

        self.view_price_series = filter_series(self.last_price_series)
        self.view_temp_series = filter_series(self.last_temp_series)

        # UI neu rendern
        self.clear_tables()
        for label, rows in self.view_per_route.items():
            self.add_table(f"Top {self.topN.value()} – {label} (gefiltert)", rows)
        if self.view_combined:
            self.add_table("GESAMT-RANKING (gefiltert)", self.view_combined)

        self.last_color_map = self.priceChart.plot_routes(self.view_price_series, self.last_color_map)
        # Heatmap: filtere heatmap data auf Datum
        filtered_heatmap = {}
        for lab, mp in self.last_temp_heatmap.items():
            filtered_heatmap[lab] = {d: t for d, t in mp.items() if fstart <= d <= fend}
        self.tempChart.plot_heatmap(filtered_heatmap, self.last_color_map)



    def add_table(self, title: str, rows: List[Candidate]):
        label = QLabel(title)
        label.setStyleSheet("font-weight:600; margin-top:8px;")
        self.results_layout.addWidget(label)

        table = QTableWidget()
        table.setColumnCount(8)
        table.setHorizontalHeaderLabels([
            "Total", "Out-Date", "Ret-Date",
            "Out € / Zeit", "Ret € / Zeit",
            "Out Temp LY", "Ret Temp LY",
            "Route"
        ])
        table.verticalHeader().setVisible(False)
        table.setEditTriggers(QTableWidget.NoEditTriggers)
        table.setSelectionBehavior(QTableWidget.SelectRows)
        table.setRowCount(len(rows))

        for r, c in enumerate(rows):
            table.setItem(r, 0, QTableWidgetItem(f"{c.total:.2f}"))
            table.setItem(r, 1, QTableWidgetItem(c.out_day))
            table.setItem(r, 2, QTableWidgetItem(c.ret_day))
            table.setItem(r, 3, QTableWidgetItem(f"{c.out_price:.2f} | {hhmm(c.dep_o)}→{hhmm(c.arr_o)}"))
            table.setItem(r, 4, QTableWidgetItem(f"{c.ret_price:.2f} | {hhmm(c.dep_r)}→{hhmm(c.arr_r)}"))
            table.setItem(r, 5, QTableWidgetItem(fmt_temp(c.out_temp_ly)))
            table.setItem(r, 6, QTableWidgetItem(fmt_temp(c.ret_temp_ly)))
            table.setItem(r, 7, QTableWidgetItem(c.route_label))

        table.resizeColumnsToContents()
        self.results_layout.addWidget(table)

    def on_search(self):
        start_qd: QDate = self.dStart.date()
        end_qd: QDate = self.dEnd.date()
        start_date = date(start_qd.year(), start_qd.month(), start_qd.day())
        end_date = date(end_qd.year(), end_qd.month(), end_qd.day())
        if end_date < start_date:
            QMessageBox.warning(self, "Eingabe", "End-Tag muss ≥ Start-Tag sein.")
            return

        if self.maxDays.value() < self.minDays.value():
            QMessageBox.warning(self, "Eingabe", "Max. Tage muss ≥ Min. Tage sein.")
            return

        idx = self.mode.currentIndex()
        routes: List[Tuple[str, str]] = []
        if idx == 0:
            routes = [("CGN", "PMO"), ("NRN", "TPS")]
        elif idx == 1:
            routes = [(self.o1.text().upper(), self.d1.text().upper()),
                      (self.o2.text().upper(), self.d2.text().upper())]
        else:
            routes = [(self.o1.text().upper(), self.d1.text().upper())]

        params = dict(
            currency=self.currency.currentText(),
            min_days=self.minDays.value(),
            max_days=self.maxDays.value(),
            top_n=self.topN.value(),
            routes=routes,
            start_date=start_date,
            end_date=end_date,

            use_time=self.cbUseTime.isChecked(),
            out_dep_min_h=int(self.spOutDepMin.value()),
            out_dep_max_h=int(self.spOutDepMax.value()),
            ret_dep_min_h=int(self.spRetDepMin.value()),
            ret_dep_max_h=int(self.spRetDepMax.value()),

            use_weather_filter=self.cbUseWeatherFilter.isChecked(),
            require_weather=self.cbRequireWeather.isChecked(),
            ideal_temp=float(self.spIdealTemp.value()),
            temp_tol=float(self.spTempTol.value()),
        )

        self.btnSearch.setEnabled(False)
        self.statusBar().showMessage("Suche läuft… bitte warten.")
        QApplication.setOverrideCursor(Qt.WaitCursor)
        self.clear_tables()
        self.priceChart.plot_routes({})
        self.tempChart.plot_heatmap({})

        self.thread = QThread()
        self.worker = Worker(params)
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.finished.connect(self.on_worker_finished)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)
        self.thread.start()

    def on_worker_finished(self, per_route, combined, price_series, temp_series, temp_heatmap, error):
        self.btnSearch.setEnabled(True)
        self.statusBar().showMessage("Fertig.")
        QApplication.restoreOverrideCursor()
        if error:
            self.statusBar().showMessage("Abgebrochen (Fehler).")
            QApplication.restoreOverrideCursor()
            QMessageBox.critical(self, "Fehler", error or "Unbekannter Fehler")
            return

        for label, rows in per_route.items():
            self.add_table(f"Top {self.topN.value()} – {label}", rows)
        if combined:
            self.add_table("GESAMT-RANKING", combined)

        self.priceChart.plot_routes(price_series)
        self.tempChart.plot_heatmap(temp_heatmap)

        # speichern für Filter/Export
        self.last_per_route = per_route
        self.last_combined = combined
        self.last_price_series = price_series
        self.last_temp_series = temp_series
        self.last_temp_heatmap = temp_heatmap

        self.btnExportCSV.setEnabled(True)
        self.btnExportPNG.setEnabled(True)


        # Slider initialisieren auf den Suchzeitraum
        self.base_start_date = self.dStart.date().toPython()
        self.base_end_date = self.dEnd.date().toPython()
        days = (self.base_end_date - self.base_start_date).days
        if days < 0:
            days = 0

        self.sStart.setRange(0, days)
        self.sEnd.setRange(0, days)
        self.sStart.setValue(0)
        self.sEnd.setValue(days)
        self.sStart.setEnabled(True)
        self.sEnd.setEnabled(True)

        # initialer Filter = alles
        self.on_filter_changed()
        return

    def export_csv(self):
        path, _ = QFileDialog.getSaveFileName(self, "CSV speichern", "roundtrips.csv", "CSV (*.csv)")
        if not path:
            return

        # exportiere aktuell gefilterte Ansicht
        rows = []
        for label, items in self.view_per_route.items():
            rows.extend(items)
        if self.view_combined:
            rows = self.view_combined  # lieber das Ranking exportieren

        import csv
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["total", "out_day", "ret_day", "out_price", "ret_price", "out_dep", "out_arr", "ret_dep", "ret_arr", "out_temp_ly", "ret_temp_ly", "route"])
            for c in rows:
                w.writerow([
                    f"{c.total:.2f}",
                    c.out_day, c.ret_day,
                    f"{c.out_price:.2f}", f"{c.ret_price:.2f}",
                    c.dep_o or "", c.arr_o or "",
                    c.dep_r or "", c.arr_r or "",
                    "" if c.out_temp_ly is None else f"{c.out_temp_ly:.1f}",
                    "" if c.ret_temp_ly is None else f"{c.ret_temp_ly:.1f}",
                    c.route_label
                ])

        QMessageBox.information(self, "Export", f"CSV gespeichert:\n{path}")

    def export_png(self):
        path, _ = QFileDialog.getSaveFileName(self, "PNG speichern", "charts.png", "PNG (*.png)")
        if not path:
            return

        base = path[:-4] if path.lower().endswith(".png") else path
        p1 = base + "_prices.png"
        p2 = base + "_weather.png"

        self.priceChart.figure.savefig(p1, dpi=150)
        self.tempChart.figure.savefig(p2, dpi=150)

        QMessageBox.information(self, "Export", f"PNG gespeichert:\n{p1}\n{p2}")

    def on_clear_cache_ryr(self):
        n = clear_cache("ryanair")
        QMessageBox.information(self, "Cache", f"Flug-Cache geleert: {n} Datei(en).")

    def on_clear_cache_weather(self):
        n = clear_cache("weather")
        QMessageBox.information(self, "Cache", f"Wetter-Cache geleert: {n} Datei(en).")

    def on_clear_cache_all(self):
        n = clear_cache("all")
        QMessageBox.information(self, "Cache", f"Alle Caches geleert: {n} Datei(en).")



def main():
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()