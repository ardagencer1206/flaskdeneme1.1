# -*- coding: utf-8 -*-
import os
import json
import traceback
from typing import Dict, Any, List, Tuple
from pathlib import Path

from flask import Flask, request, jsonify, send_from_directory

from pyomo.environ import (
    ConcreteModel, Set, Param, Var, Binary, NonNegativeReals, Objective, minimize,
    Constraint, Any as PyAny, value, SolverFactory
)
from pyomo.opt import TerminationCondition

# =======================
# Konfig
# =======================
DATASET_PATH = Path(os.environ.get("DATASET_PATH", str(Path(__file__).parent / "dataset.json")))
MAX_SOLVE_SECONDS = 26

# (Opsiyonel) Azure OpenAI Chat (UI'deki Chat için)
from openai import AzureOpenAI
AZURE_OPENAI_API_KEY = os.environ.get("AZURE_OPENAI_API_KEY", "d0167637046c4443badc4920cc612abb")
AZURE_OPENAI_ENDPOINT = os.environ.get("AZURE_OPENAI_ENDPOINT", "https://openai-fnss.openai.azure.com")
AZURE_DEPLOYMENT_NAME = os.environ.get("AZURE_DEPLOYMENT_NAME", "gpt-4o")
AZURE_OPENAI_API_VERSION = os.environ.get("AZURE_OPENAI_API_VERSION", "2024-06-01")

try:
    aoai_client = AzureOpenAI(
        api_key=AZURE_OPENAI_API_KEY,
        api_version=AZURE_OPENAI_API_VERSION,
        azure_endpoint=AZURE_OPENAI_ENDPOINT
    ) if AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT else None
except Exception:
    aoai_client = None

app = Flask(__name__, static_url_path="", static_folder=".")

# =======================
# Yardımcılar
# =======================
def safe_float(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return default

def load_dataset_from_file() -> Dict[str, Any]:
    """dataset.json'u oku ve zorunlu alanları kontrol et."""
    try:
        with open(DATASET_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        required = [
            "cities", "main_depot", "periods",
            "vehicle_types", "vehicle_count",
            "distances", "packages", "minutil_penalty"
        ]
        for k in required:
            if k not in data:
                raise KeyError(f"dataset.json alan eksik: {k}")
        return data
    except FileNotFoundError:
        raise FileNotFoundError(f"dataset.json bulunamadı: {DATASET_PATH}")
    except Exception as e:
        raise RuntimeError(f"dataset.json okunamadı: {e}")

def pick_solver():
    """Önce APPsi-HiGHS; sonra highs/cbc/glpk/cplex."""
    try:
        from pyomo.contrib.appsi.solvers.highs import Highs as AppsiHighs
        s = AppsiHighs()
        try:
            s.config.time_limit = MAX_SOLVE_SECONDS
        except Exception:
            pass
        return "appsi_highs", s, True
    except Exception:
        pass

    for cand in ["highs", "cbc", "glpk", "cplex"]:
        try:
            s = SolverFactory(cand)
            if s is not None and s.available():
                try:
                    if cand == "highs":
                        s.options["time_limit"] = MAX_SOLVE_SECONDS
                    elif cand == "cbc":
                        s.options["seconds"] = int(MAX_SOLVE_SECONDS)
                    elif cand == "glpk":
                        s.options["tmlim"] = int(MAX_SOLVE_SECONDS)
                    elif cand == "cplex":
                        s.options["timelimit"] = MAX_SOLVE_SECONDS
                        s.options["mipgap"] = 0.05
                        s.options["threads"] = 2
                except Exception:
                    pass
                return cand, s, False
        except Exception:
            continue
    return None, None, False

# =======================
# Model
# =======================
def build_model(payload: Dict[str, Any]) -> Tuple[ConcreteModel, Dict[str, Any]]:
    """
    periods tabanlı zaman adımı (τ=1). Her t'de:
      - Araç tek şehirde (loc[v,n,t]) ve en fazla 1 hareket (x[v,i,j,t])
      - Paket tek şehirde (pkg_loc[p,n,t]); y[p,v,i,j,t] varsa akış ve konum geçişi yapılır
    Böylece kopuk yol oluşmaz; araç ve paket zaman çizelgesi tutarlıdır.
    """
    # Girdi
    cities: List[str] = payload["cities"]
    main_depot: str = payload["main_depot"]
    periods: int = int(payload["periods"])
    Tmin, Tmax = 1, periods
    T = list(range(Tmin, Tmax + 1))

    vtypes: Dict[str, Dict[str, Any]] = payload["vehicle_types"]
    vcount: Dict[str, int] = payload["vehicle_count"]
    vehicles: List[str] = [f"{vt}_{i}" for vt, cnt in vcount.items() for i in range(1, int(cnt) + 1)]

    # Mesafeler (simetrik)
    distances: Dict[Tuple[str, str], float] = {}
    for i, j, d in payload["distances"]:
        distances[(i, j)] = float(d)
        distances[(j, i)] = float(d)
    for c in cities:
        distances[(c, c)] = 0.0

    # Paketler
    packages_in: List[Dict[str, Any]] = payload["packages"]
    packages = {}
    for rec in packages_in:
        pid = str(rec["id"])
        # edit.html'den gelen son formatı da destekleyelim
        ready = int(rec.get("ready", rec.get("ready_hour", 0)))
        dline = int(rec.get("deadline_suresi", rec.get("deadline_hour", 36)))
        packages[pid] = {
            "o": rec["baslangic"],
            "d": rec["hedef"],
            "w": float(rec["agirlik"]),
            "ready": ready if ready >= Tmin else Tmin,
            "deadline": dline,
            "pen": float(rec.get("ceza", 0)),
        }

    MINUTIL_PENALTY = safe_float(payload.get("minutil_penalty", 10.0), 10.0)

    # Pyomo
    m = ConcreteModel()
    m.Cities  = Set(initialize=cities)
    m.Periods = Set(initialize=T)
    m.Vehicles = Set(initialize=vehicles)
    m.Packages = Set(initialize=list(packages.keys()))

    def vtype(v): return v.rsplit("_", 1)[0]

    # Parametreler
    m.Distance = Param(m.Cities, m.Cities, initialize=lambda _m, i, j: distances[(i, j)])
    m.VehicleCapacity = Param(m.Vehicles, initialize=lambda _m, v: vtypes[vtype(v)]["kapasite"])
    m.TransportCost   = Param(m.Vehicles, initialize=lambda _m, v: vtypes[vtype(v)]["maliyet_km"])
    m.FixedCost       = Param(m.Vehicles, initialize=lambda _m, v: vtypes[vtype(v)]["sabit_maliyet"])
    m.MinUtilization  = Param(m.Vehicles, initialize=lambda _m, v: vtypes[vtype(v)]["min_doluluk"])

    m.PW = Param(m.Packages,  initialize=lambda _m, p: packages[p]["w"])
    m.Org= Param(m.Packages, within=PyAny, initialize=lambda _m, p: packages[p]["o"])
    m.Des= Param(m.Packages, within=PyAny, initialize=lambda _m, p: packages[p]["d"])
    m.Rdy= Param(m.Packages,  initialize=lambda _m, p: packages[p]["ready"])
    m.Dln= Param(m.Packages,  initialize=lambda _m, p: packages[p]["deadline"])
    m.Pen= Param(m.Packages,  initialize=lambda _m, p: packages[p]["pen"])

    # Değişkenler
    m.x = Var(m.Vehicles, m.Cities, m.Cities, m.Periods, domain=Binary)          # araç hareketi
    m.y = Var(m.Packages, m.Vehicles, m.Cities, m.Cities, m.Periods, domain=Binary)  # paket hareketi
    m.z = Var(m.Vehicles, m.Periods, domain=Binary)                               # araç kullanımı
    m.loc = Var(m.Vehicles, m.Cities, m.Periods, domain=Binary)                   # araç konumu
    m.pkg_loc = Var(m.Packages, m.Cities, m.Periods, domain=Binary)               # paket konumu
    m.late = Var(m.Packages, domain=NonNegativeReals)                             # gecikme
    m.minutil_short = Var(m.Vehicles, m.Periods, domain=NonNegativeReals)         # doluluk açığı

    # Amaç
    def obj_rule(_m):
        transport = sum(_m.TransportCost[v] * _m.Distance[i, j] * _m.x[v, i, j, t]
                        for v in _m.Vehicles for i in _m.Cities for j in _m.Cities for t in _m.Periods if i != j)
        fixed = sum(_m.FixedCost[v] * _m.z[v, t] for v in _m.Vehicles for t in _m.Periods)
        late = sum(_m.Pen[p] * _m.late[p] for p in _m.Packages)
        minutil = MINUTIL_PENALTY * sum(_m.minutil_short[v, t] for v in _m.Vehicles for t in _m.Periods)
        return transport + fixed + late + minutil
    m.obj = Objective(rule=obj_rule, sense=minimize)

    # ---------- Kısıtlar ----------
    # (1) Paket origin'den tek çıkış (ready'den sonra)
    def pkg_origin(_m, p):
        o, r = _m.Org[p], _m.Rdy[p]
        return sum(_m.y[p, v, o, j, t] for v in _m.Vehicles for j in _m.Cities for t in _m.Periods if j != o and t >= r) == 1
    m.pkg_origin = Constraint(m.Packages, rule=pkg_origin)

    # (2) Paket hedefe tek varış
    def pkg_dest(_m, p):
        d = _m.Des[p]
        return sum(_m.y[p, v, i, d, t] for v in _m.Vehicles for i in _m.Cities for t in _m.Periods if i != d) == 1
    m.pkg_dest = Constraint(m.Packages, rule=pkg_dest)

    # (3) Paket->Araç bağlantısı (y <= x)
    def link_yx(_m, p, v, i, j, t):
        if i == j: return Constraint.Skip
        return _m.y[p, v, i, j, t] <= _m.x[v, i, j, t]
    m.link_yx = Constraint(m.Packages, m.Vehicles, m.Cities, m.Cities, m.Periods, rule=link_yx)

    # (4) Kapasite
    def cap(_m, v, i, j, t):
        if i == j: return Constraint.Skip
        return sum(_m.PW[p] * _m.y[p, v, i, j, t] for p in _m.Packages) <= _m.VehicleCapacity[v]
    m.cap = Constraint(m.Vehicles, m.Cities, m.Cities, m.Periods, rule=cap)

    # (5) Min. doluluk (sadece ana depodan çıkışlar)
    def minutil(_m, v, t):
        departures = sum(_m.x[v, main_depot, j, t] for j in _m.Cities if j != main_depot)
        loaded = sum(_m.PW[p] * _m.y[p, v, main_depot, j, t] for p in _m.Packages for j in _m.Cities if j != main_depot)
        target = _m.MinUtilization[v] * _m.VehicleCapacity[v] * departures
        return loaded + _m.minutil_short[v, t] >= target
    m.minutil = Constraint(m.Vehicles, m.Periods, rule=minutil)

    # (6) Paket konumu: her t'de tek şehir
    m.pkg_onehot = Constraint(m.Packages, m.Periods,
        rule=lambda _m, p, t: sum(_m.pkg_loc[p, n, t] for n in _m.Cities) == 1)

    # (7) Ready öncesi origin’de kilit
    def pkg_before_ready(_m, p, n, t):
        o, r = _m.Org[p], _m.Rdy[p]
        if t < r:
            return _m.pkg_loc[p, n, t] == (1 if n == o else 0)
        return Constraint.Skip
    m.pkg_before_ready = Constraint(m.Packages, m.Cities, m.Periods, rule=pkg_before_ready)

    # (8) Paket konum geçişi (τ=1)
    def pkg_trans(_m, p, n, t):
        if t == Tmax: return Constraint.Skip
        incoming = sum(_m.y[p, v, i, n, t] for v in _m.Vehicles for i in _m.Cities if i != n)
        outgoing = sum(_m.y[p, v, n, j, t] for v in _m.Vehicles for j in _m.Cities if j != n)
        return _m.pkg_loc[p, n, t] + incoming - outgoing == _m.pkg_loc[p, n, t+1]
    m.pkg_trans = Constraint(m.Packages, m.Cities, m.Periods, rule=pkg_trans)

    # (9) Paket ancak bulunduğu şehirden çıkabilir
    def pkg_depart_possible(_m, p, i, t):
        return sum(_m.y[p, v, i, j, t] for v in _m.Vehicles for j in _m.Cities if j != i) <= _m.pkg_loc[p, i, t]
    m.pkg_depart_possible = Constraint(m.Packages, m.Cities, m.Periods, rule=pkg_depart_possible)

    # (10) Paket varış -> t+1’de o şehirde olmalı
    def pkg_arrive_possible(_m, p, j, t):
        if t == Tmax: return Constraint.Skip
        return sum(_m.y[p, v, i, j, t] for v in _m.Vehicles for i in _m.Cities if i != j) <= _m.pkg_loc[p, j, t+1]
    m.pkg_arrive_possible = Constraint(m.Packages, m.Cities, m.Periods, rule=pkg_arrive_possible)

    # (11) Araç: her t’de tek şehir
    m.veh_onehot = Constraint(m.Vehicles, m.Periods,
        rule=lambda _m, v, t: sum(_m.loc[v, n, t] for n in _m.Cities) == 1)

    # (12) Araç başlangıç konumu (t=1: ana depo)
    m.veh_init = Constraint(m.Vehicles, rule=lambda _m, v: _m.loc[v, main_depot, Tmin] == 1)

    # (13) Araç konum geçişi (τ=1)
    def veh_trans(_m, v, n, t):
        if t == Tmax: return Constraint.Skip
        incoming = sum(_m.x[v, i, n, t] for i in _m.Cities if i != n)
        outgoing = sum(_m.x[v, n, j, t] for j in _m.Cities if j != n)
        return _m.loc[v, n, t] + incoming - outgoing == _m.loc[v, n, t+1]
    m.veh_trans = Constraint(m.Vehicles, m.Cities, m.Periods, rule=veh_trans)

    # (14) Araç bulunduğu şehirden ayrılabilir
    def veh_move_from_loc(_m, v, i, t):
        return sum(_m.x[v, i, j, t] for j in _m.Cities if j != i) <= _m.loc[v, i, t]
    m.veh_move_from_loc = Constraint(m.Vehicles, m.Cities, m.Periods, rule=veh_move_from_loc)

    # (15) Araç periyot başına en fazla 1 hareket
    m.veh_one_move = Constraint(m.Vehicles, m.Periods,
        rule=lambda _m, v, t: sum(_m.x[v, i, j, t] for i in _m.Cities for j in _m.Cities if i != j) <= 1)

    # (16) Araç kullanımı bayrağı
    m.veh_used = Constraint(m.Vehicles, m.Periods,
        rule=lambda _m, v, t: _m.z[v, t] >= sum(_m.x[v, i, j, t] for i in _m.Cities for j in _m.Cities if i != j))

    # (17) Gecikme tanımı (teslim_t - (ready+deadline) <= late)
    def lateness(_m, p):
        d = _m.Des[p]
        deliver_t = sum(t * _m.y[p, v, i, d, t] for v in _m.Vehicles for i in _m.Cities for t in _m.Periods if i != d)
        return _m.late[p] >= deliver_t - (_m.Rdy[p] + _m.Dln[p])
    m.lateness = Constraint(m.Packages, rule=lateness)

    # (18) Aynı paketin aynı i->j segmentini toplamda en fazla 1 kez kullanması (döngüyü azaltır)
    def pkg_once_seg(_m, p, i, j):
        if i == j: return Constraint.Skip
        return sum(_m.y[p, v, i, j, t] for v in _m.Vehicles for t in _m.Periods) <= 1
    m.pkg_once_seg = Constraint(m.Packages, m.Cities, m.Cities, rule=pkg_once_seg)

    meta = {
        "cities": cities,
        "periods": T,
        "vehicles": vehicles,
        "packages": packages,   # {id: {o,d,w,ready,deadline,pen}}
        "distances": distances,
        "vtypes": vtypes,
        "main_depot": main_depot,
        "MINUTIL_PENALTY": MINUTIL_PENALTY
    }
    return m, meta

# =======================
# Sonuç çıkarımı
# =======================
def extract_results(model: ConcreteModel, meta: Dict[str, Any]) -> Dict[str, Any]:
    cities = meta["cities"]; T = meta["periods"]; vehicles = meta["vehicles"]
    packages = meta["packages"]; distances = meta["distances"]; MINUTIL = meta["MINUTIL_PENALTY"]

    out: Dict[str, Any] = {}
    out["objective"] = float(value(model.obj))

    # Maliyetler
    transport = 0.0
    for v in vehicles:
        for i in cities:
            for j in cities:
                for t in T:
                    if i != j and value(model.x[v, i, j, t]) > 0.5:
                        transport += float(value(model.TransportCost[v])) * float(value(model.Distance[i, j]))
    fixed = sum(float(value(model.FixedCost[v])) for v in vehicles for t in T if value(model.z[v, t]) > 0.5)
    late = sum(float(value(model.Pen[p])) * float(value(model.late[p])) for p in model.Packages)
    minutil_pen = MINUTIL * sum(float(value(model.minutil_short[v, t])) for v in vehicles for t in T)
    out["cost_breakdown"] = {
        "transport": transport, "fixed": fixed, "lateness": late, "min_util_gap": minutil_pen
    }

    # Araç rotaları (zaman sıralı)
    v_routes = []
    for v in sorted(vehicles):
        legs = []
        for t in T:
            for i in cities:
                for j in cities:
                    if i != j and value(model.x[v, i, j, t]) > 0.5:
                        moved = []
                        totw = 0.0
                        for p in model.Packages:
                            if value(model.y[p, v, i, j, t]) > 0.5:
                                moved.append(p)
                                totw += float(value(model.PW[p]))
                        legs.append({
                            "t": t, "from": i, "to": j,
                            "km": float(distances[(i, j)]),
                            "packages": moved,
                            "load_kg": totw,
                            "utilization_pct": (100.0 * totw / float(value(model.VehicleCapacity[v]))) if totw>0 else 0.0
                        })
        if legs:
            v_routes.append({"vehicle": v, "capacity": float(value(model.VehicleCapacity[v])), "legs": legs})
    out["vehicle_routes"] = v_routes

    # Paketler + rotaları (zaman sıralı)
    p_summ = []
    for p in sorted(packages.keys()):
        o, d = packages[p]["o"], packages[p]["d"]
        r, dl = packages[p]["ready"], packages[p]["deadline"]
        deadline_at = r + dl

        delivered_t = None
        for t in T:
            if sum(value(model.y[p, v, i, d, t]) for v in model.Vehicles for i in model.Cities if i != d) > 0.5:
                delivered_t = t
                break

        segs = []
        for t in T:
            for v in model.Vehicles:
                for i in cities:
                    for j in cities:
                        if i != j and value(model.y[p, v, i, j, t]) > 0.5:
                            segs.append({"t": t, "from": i, "to": j, "vehicle": v, "km": float(distances[(i, j)])})

        p_summ.append({
            "id": p,
            "origin": o, "dest": d,
            "weight": packages[p]["w"],
            "ready": r,
            "deadline_by": deadline_at,
            "delivered_at": delivered_t,
            "on_time": (delivered_t is not None and delivered_t <= deadline_at),
            "lateness_hours": (max(0, delivered_t - deadline_at) if delivered_t is not None else None),
            "lateness_penalty": float(value(model.Pen[p])) * (max(0, delivered_t - deadline_at) if delivered_t is not None else 0.0),
            "route": segs
        })
    out["packages"] = p_summ
    return out

# =======================
# HTTP Routes
# =======================
@app.route("/")
def root():
    return send_from_directory(".", "index.html")

@app.route("/dataset", methods=["GET", "PUT", "POST"])
def dataset_endpoint():
    try:
        if request.method == "GET":
            if not DATASET_PATH.exists():
                return jsonify({"ok": False, "error": "dataset.json bulunamadı"}), 404
            with open(DATASET_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            return jsonify({"ok": True, "dataset": data})

        # PUT/POST -> kaydet
        payload = request.get_json(force=True)
        required = ["cities","main_depot","periods","vehicle_types","vehicle_count","distances","packages","minutil_penalty"]
        miss = [k for k in required if k not in payload]
        if miss:
            return jsonify({"ok": False, "error": f"Eksik alanlar: {', '.join(miss)}"}), 400

        tmp = DATASET_PATH.with_suffix(".json.tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        tmp.replace(DATASET_PATH)
        return jsonify({"ok": True, "message": "dataset.json güncellendi"})
    except Exception as e:
        return jsonify({"ok": False, "error": f"Hata: {e}"}), 500

@app.route("/chat", methods=["POST"])
def chat():
    try:
        if aoai_client is None:
            return jsonify({"ok": False, "error": "Azure OpenAI istemcisi yok veya yapılandırılmadı."}), 500
        payload = request.get_json(force=True) or {}
        user_messages = payload.get("messages", [])
        ctx = payload.get("context", {})

        sys_prompt = f"""
Sen bir lojistik optimizasyon asistanısın. Kullanıcıdan gelen VRP parametrelerini
(şehirler, dönemler, ana depo, araç tip/sayıları, mesafeler, paketler, min. doluluk)
kullanarak kısa ve net yanıt ver. 
Model JSON:
{ctx}
"""
        comp = aoai_client.chat.completions.create(
            model=AZURE_DEPLOYMENT_NAME,
            messages=[{"role":"system","content":sys_prompt}, *user_messages],
            temperature=0.2, max_tokens=600
        )
        return jsonify({"ok": True, "answer": comp.choices[0].message.content})
    except Exception as e:
        return jsonify({"ok": False, "error": f"Hata: {e}", "trace": traceback.format_exc()}), 500

@app.route("/health")
def health():
    return jsonify({"ok": True})

# =======================
# Solve
# =======================
@app.route("/solve", methods=["POST"])
def solve():
    try:
        data = request.get_json(silent=True) or {}
        if not data:
            data = load_dataset_from_file()

        model, meta = build_model(data)
        solver_name, solver, is_appsi = pick_solver()
        if solver is None:
            return jsonify({"ok": False, "error": "Uygun MILP çözücüsü bulunamadı."}), 400

        if is_appsi:
            results = solver.solve(model)
            term = getattr(results, "termination_condition", None)
        else:
            try:
                results = solver.solve(model, tee=False, load_solutions=True)
            except TypeError:
                results = solver.solve(model, load_solutions=True)
            term = getattr(getattr(results, "solver", None), "termination_condition", None) or getattr(results, "termination_condition", None)

        def has_incumbent(m: ConcreteModel) -> bool:
            try:
                for _, v in m.x.items():
                    if v.value is not None:
                        return True
            except Exception:
                pass
            return False

        diag = {
            "termination": str(term),
            "solver": solver_name,
            "wallclock_time": getattr(getattr(results, "solver", None), "wallclock_time", None),
            "gap": getattr(getattr(results, "solver", None), "gap", None),
            "status": getattr(getattr(results, "solver", None), "status", None),
        }

        if has_incumbent(model):
            out = extract_results(model, meta)
            return jsonify({"ok": True, "solver": solver_name, "result": out, "diagnostics": diag})

        return jsonify({
            "ok": False,
            "error": f"{MAX_SOLVE_SECONDS}s içinde uygulanabilir çözüm bulunamadı. Durum: {term}",
            "diagnostics": diag
        }), 200

    except Exception as e:
        return jsonify({"ok": False, "error": f"Hata: {e}", "trace": traceback.format_exc()}), 500

@app.errorhandler(500)
def handle_500(e):
    return jsonify({"ok": False, "error": "Internal Server Error"}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

