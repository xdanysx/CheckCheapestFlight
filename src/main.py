# -*- coding: utf-8 -*-
import sys
import requests
import calendar
from datetime import datetime, timedelta, date
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from pathlib import Path

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
    """
    Holt fuer Monat y-m die cheapestPerDay-Daten.
    Rueckgabe: dict[YYYY-MM-DD] = { price: float, dep: str|None, arr: str|None }
    """
    cfg = load_api_config()
    url_template = cfg["RYANAIR_API_URL_TEMPLATE"]

    month_str = f"{y:04d}-{m:02d}-01"
    url = url_template.format(origin_iata, dest_iata)
    params = {"outboundMonthOfDate": month_str, "currency": curr}

    try:
        r = requests.get(url, params=params, headers=HEADERS, timeout=20)
        r.raise_for_status()
        data = r.json()
    except requests.RequestException:
        return {}
    except ValueError:
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

    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": ly.isoformat(),
        "end_date": ly.isoformat(),
        "daily": ["temperature_2m_max"],  # <-- als LISTE!
        "timezone": "UTC",                # <-- WICHTIG
    }

    try:
        r = requests.get(url, params=params, timeout=20)
        r.raise_for_status()
        j = r.json()


        daily = j.get("daily")
        if not daily:
            return None

        temps = daily.get("temperature_2m_max")
        if not temps:
            return None

        t = temps[0]
        if isinstance(t, (int, float)):
            return float(t)

        return None
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
    QTabWidget, QScrollArea, QDateEdit, QFileDialog, QSlider
)

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.dates as mdates


def fmt_temp(t: Optional[float]) -> str:
    return "-" if t is None else f"{t:.1f}°C"


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


class TempChart(FigureCanvas):
    def __init__(self, parent=None):
        fig = Figure(figsize=(7, 4), tight_layout=True)
        self.ax = fig.add_subplot(111)
        super().__init__(fig)
        self.setParent(parent)

    def plot_routes(self, temp_series: Dict[str, List[Tuple[date, float]]], color_map: Optional[Dict[str, str]] = None):
        self.ax.clear()
        if color_map is None:
            color_map = {}

        for label, data in temp_series.items():
            if not data:
                continue
            data_sorted = sorted(data, key=lambda t: t[0])
            xs = [d for d, _ in data_sorted]
            ys = [t for _, t in data_sorted]

            self.ax.plot(xs, ys, linestyle="solid", label=label, color=color_map.get(label))

        self.ax.set_title("Temperatur (letztes Jahr) am Ziel pro Abflugtag")
        self.ax.set_xlabel("Abflugdatum")
        self.ax.set_ylabel("Temp (°C, LY)")
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

class Worker(QObject):
    finished = Signal(dict, list, dict, dict, str)

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

        month_cache: Dict[Tuple[str, str, int, int, str], Dict[str, DayQuote]] = {}
        per_route: Dict[str, List[Candidate]] = {}
        all_cands: List[Candidate] = []

        price_series: Dict[str, List[Tuple[date, float]]] = {}
        temp_series: Dict[str, List[Tuple[date, float]]] = {}

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

        try:
            start_date: date = p["start_date"]
            end_date: date = p["end_date"]

            for (o, d) in routes:
                label = f"{o} ↔ {d}"
                cands = find_roundtrips_for_route_by_dates(
                    o, d, start_date, end_date, min_days, max_days, currency, month_cache
                )

                rows = cands[:top_n]
                for c in rows:
                    # "da wo man fliegt" -> Ziel (dest)
                    c.out_temp_ly = get_temp_ly_for_iata(c.dest_iata, c.out_day)
                    c.ret_temp_ly = get_temp_ly_for_iata(c.dest_iata, c.ret_day)
                per_route[label] = rows

                all_cands.extend(cands)

                best_per_day: Dict[str, float] = {}
                for c in cands:
                    best_per_day[c.out_day] = min(best_per_day.get(c.out_day, float("inf")), c.total)

                p_points: List[Tuple[date, float]] = []
                t_points: List[Tuple[date, float]] = []

                for day_str, best_total in best_per_day.items():
                    try:
                        dt_day = datetime.strptime(day_str, "%Y-%m-%d").date()
                    except ValueError:
                        continue
                    p_points.append((dt_day, best_total))

                    t = get_temp_ly_for_iata(d, day_str)
                    if t is not None:
                        t_points.append((dt_day, t))

                price_series[label] = p_points
                temp_series[label] = t_points

            if len(routes) >= 2 and all_cands:
                all_cands.sort(key=lambda x: (x.total, x.out_day))
                combined = all_cands[:top_n]
                for c in combined:
                    c.out_temp_ly = get_temp_ly_for_iata(c.dest_iata, c.out_day)
                    c.ret_temp_ly = get_temp_ly_for_iata(c.dest_iata, c.ret_day)

        except Exception as e:
            error = str(e)

        # Wenn im Plot nirgendwo Temperaturen ankamen, sag es klar (aber ohne crash)
        # 1) Wenn es gar keine Preis-Punkte gibt -> keine Flüge gefunden
        if all((not pts) for pts in price_series.values()):
            error = "Keine Flüge/Preise in diesem Zeitraum gefunden (Ryanair API hat keine Daten geliefert)."
        # 2) Flüge existieren, aber Temp-Serie leer -> Wetterproblem
        elif all((not pts) for pts in temp_series.values()):
            error = "Flüge gefunden, aber keine Temperaturdaten (Open-Meteo). Prüfe Internet/Firewall oder versuche kürzeren Zeitraum."



        self.finished.emit(per_route, combined, price_series, temp_series, error)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.last_per_route = {}
        self.last_combined = []
        self.last_price_series = {}
        self.last_temp_series = {}
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
        self.tempChart = TempChart()
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

        g_time = QGroupBox("Zeitraum & Optionen")
        gt = QGridLayout(g_time)
        gt.addWidget(QLabel("Start-Tag"), 0, 0); gt.addWidget(self.dStart, 0, 1)
        gt.addWidget(QLabel("End-Tag"), 0, 2);   gt.addWidget(self.dEnd, 0, 3)
        gt.addWidget(QLabel("Min. Tage"), 1, 0); gt.addWidget(self.minDays, 1, 1)
        gt.addWidget(QLabel("Max. Tage"), 1, 2); gt.addWidget(self.maxDays, 1, 3)
        gt.addWidget(QLabel("Top-N"), 1, 4);     gt.addWidget(self.topN, 1, 5)
        gt.addWidget(QLabel("Währung"), 2, 0);   gt.addWidget(self.currency, 2, 1)
        left.addWidget(g_time)

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
        self.tempChart.plot_routes(self.view_temp_series, self.last_color_map)


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
        )

        self.btnSearch.setEnabled(False)
        self.clear_tables()
        self.priceChart.plot_routes({})
        self.tempChart.plot_routes({})


        self.thread = QThread()
        self.worker = Worker(params)
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.finished.connect(self.on_worker_finished)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)
        self.thread.start()

    def on_worker_finished(self, per_route: dict, combined: list, price_series: dict, temp_series: dict, error: str):
        self.btnSearch.setEnabled(True)
        if error:
            QMessageBox.critical(self, "Fehler", error or "Unbekannter Fehler")
            return

        for label, rows in per_route.items():
            self.add_table(f"Top {self.topN.value()} – {label}", rows)
        if combined:
            self.add_table("GESAMT-RANKING", combined)

        self.priceChart.plot_routes(price_series)
        self.tempChart.plot_routes(temp_series)

        # speichern für Filter/Export
        self.last_per_route = per_route
        self.last_combined = combined
        self.last_price_series = price_series
        self.last_temp_series = temp_series

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



def main():
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
