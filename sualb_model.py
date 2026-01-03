"""
Parser und Gurobi-Modell für das Setup Assembly Line Balancing and Scheduling Problem (SUALBSP-I).

Das Skript liest .alb-Dateien aus dem Datenordner, erzeugt ein Datenobjekt und baut
ein Gurobi-Modell, das die Anzahl der genutzten Stationen minimiert (Type-I-Variante
mit vorgegebenem Zyklus). Es unterstützt Vorwärts- und Rückwärts-Rüstzeiten gemäß
Problemdefinition.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import gurobipy as gp
from gurobipy import GRB


@dataclass
class SUALBInstance:
    n_tasks: int
    cycle_time: int
    task_times: Dict[int, int]
    precedences: List[Tuple[int, int]]
    setup_forward: Dict[Tuple[int, int], int]
    setup_backward: Dict[Tuple[int, int], int]


def _parse_section(lines: Iterable[str]) -> Dict[str, List[str]]:
    """Gruppiert Zeilen einer .alb-Datei nach Abschnittsüberschrift."""

    sections: Dict[str, List[str]] = {}
    current = None
    for raw in lines:
        line = raw.strip()
        if not line:
            continue
        if line.startswith("<") and line.endswith(">"):
            current = line.strip("<>").strip()
            sections[current] = []
            continue
        if current is None:
            continue
        sections[current].append(line)
    return sections


def parse_alb_file(path: Path) -> SUALBInstance:
    """Liest eine .alb-Datei und gibt eine SUALBInstance zurück."""

    sections = _parse_section(path.read_text().splitlines())

    n_tasks = int(sections["number of tasks"][0])
    cycle_time = int(sections["cycle time"][0])

    task_times: Dict[int, int] = {}
    for entry in sections["task times"]:
        task, duration = entry.split()
        task_times[int(task)] = int(duration)

    precedences: List[Tuple[int, int]] = []
    for entry in sections["precedence relations"]:
        i, j = entry.split(",")
        precedences.append((int(i), int(j)))

    setup_forward: Dict[Tuple[int, int], int] = {}
    for entry in sections.get("setup times forward", []):
        i, rest = entry.split(",")
        j, t = rest.split(":")
        setup_forward[(int(i), int(j))] = int(t)

    setup_backward: Dict[Tuple[int, int], int] = {}
    for entry in sections.get("setup times backward", []):
        i, rest = entry.split(",")
        j, t = rest.split(":")
        setup_backward[(int(i), int(j))] = int(t)

    return SUALBInstance(
        n_tasks=n_tasks,
        cycle_time=cycle_time,
        task_times=task_times,
        precedences=precedences,
        setup_forward=setup_forward,
        setup_backward=setup_backward,
    )


def build_sualb_model(
    instance: SUALBInstance,
    *,
    max_stations: Optional[int] = None,
    model_name: str = "SUALBSP-I",
) -> gp.Model:
    """Erzeugt ein Gurobi-Modell für SUALBSP-I (Minimierung der Stationsanzahl)."""

    tasks = list(instance.task_times.keys())
    n = instance.n_tasks
    cycle = instance.cycle_time
    ub_stations = max_stations or n
    stations = list(range(1, ub_stations + 1))

    model = gp.Model(model_name)

    # x[i,k] = 1, wenn Aufgabe i Station k zugeordnet ist
    x = model.addVars(tasks, stations, vtype=GRB.BINARY, name="x")

    # u[k] = 1, wenn Station k genutzt wird
    u = model.addVars(stations, vtype=GRB.BINARY, name="u")

    # z[i,j,k] = 1, wenn i vor j auf Station k bearbeitet wird
    z = model.addVars(tasks, tasks, stations, vtype=GRB.BINARY, name="z")

    # f[i,k], l[i,k] = 1, wenn i erste bzw. letzte Aufgabe auf Station k ist
    f = model.addVars(tasks, stations, vtype=GRB.BINARY, name="first")
    l = model.addVars(tasks, stations, vtype=GRB.BINARY, name="last")

    # b[i,j,k] = 1, wenn i letzte und j erste Aufgabe auf Station k sind (Backward-Setup)
    b = model.addVars(tasks, tasks, stations, vtype=GRB.BINARY, name="backward")

    # p[i,k] Positionsvariablen für MTZ-Ordering (1..n) auf Station k
    p = model.addVars(tasks, stations, vtype=GRB.INTEGER, lb=1, ub=n, name="pos")

    # 1) Aufgaben-Zuordnung: jede Aufgabe genau einer Station
    model.addConstrs((x.sum(i, "*") == 1 for i in tasks), name="assign_once")

    # Station genutzt, wenn eine Aufgabe zugeordnet ist
    model.addConstrs((x.sum("*", k) <= n * u[k] for k in stations), name="use_station")

    # Erste/letzte Marker nur, wenn Station genutzt
    model.addConstrs((f.sum("*", k) == u[k] for k in stations), name="one_first_per_station")
    model.addConstrs((l.sum("*", k) == u[k] for k in stations), name="one_last_per_station")
    model.addConstrs((f[i, k] <= x[i, k] for i in tasks for k in stations), name="first_only_if_assigned")
    model.addConstrs((l[i, k] <= x[i, k] for i in tasks for k in stations), name="last_only_if_assigned")

    # Ordering-Verknüpfung: z wird nur aktiv, wenn beide Aufgaben auf Station k liegen
    model.addConstrs((z[i, j, k] <= x[i, k] for i in tasks for j in tasks for k in stations), name="order_only_if_assigned_i")
    model.addConstrs((z[i, j, k] <= x[j, k] for i in tasks for j in tasks for k in stations), name="order_only_if_assigned_j")
    model.addConstrs(
        (
            z[i, j, k] + z[j, i, k] == x[i, k] + x[j, k] - (1 - u[k])
            for i in tasks for j in tasks if i < j for k in stations
        ),
        name="exact_order_if_both_on_station",
    )

    # MTZ-Ordering zur Eliminierung von Zyklen innerhalb einer Station
    big_m = n
    model.addConstrs(
        (
            p[i, k] - p[j, k] + big_m * z[i, j, k] <= big_m - 1 + (1 - x[i, k]) * big_m + (1 - x[j, k]) * big_m
            for i in tasks for j in tasks if i != j for k in stations
        ),
        name="mtz_order",
    )

    # First/last fest mit Ordering verknüpfen
    model.addConstrs(
        (z[i, j, k] >= f[i, k] + x[j, k] - 1 for i in tasks for j in tasks if i != j for k in stations),
        name="first_precedes_all",
    )
    model.addConstrs(
        (z[i, j, k] <= 1 - l[j, k] + x[i, k] for i in tasks for j in tasks if i != j for k in stations),
        name="last_followed_by_none",
    )

    # Backward-Kopplung
    model.addConstrs((b[i, j, k] <= l[i, k] for i in tasks for j in tasks for k in stations), name="backward_last")
    model.addConstrs((b[i, j, k] <= f[j, k] for i in tasks for j in tasks for k in stations), name="backward_first")
    model.addConstrs(
        (b[i, j, k] >= l[i, k] + f[j, k] - 1 for i in tasks for j in tasks for k in stations),
        name="backward_iff",
    )

    # 2) Vorgängerbeziehungen: Stationen respektieren Reihenfolge
    station_index = {i: gp.quicksum(k * x[i, k] for k in stations) for i in tasks}
    model.addConstrs((station_index[i] <= station_index[j] for i, j in instance.precedences), name="precede_station")
    model.addConstrs(
        (z[i, j, k] >= x[i, k] + x[j, k] - 1 for i, j in instance.precedences for k in stations),
        name="precede_order_same_station",
    )

    # 3) Zykluszeit je Station: Bearbeitungen + Forward-Setups + Backward-Setup <= c
    forward_default = 0
    backward_default = 0

    for k in stations:
        processing = gp.quicksum(instance.task_times[i] * x[i, k] for i in tasks)
        forward_setups = gp.quicksum(
            instance.setup_forward.get((i, j), forward_default) * z[i, j, k]
            for i in tasks
            for j in tasks
            if i != j
        )
        backward_setups = gp.quicksum(
            instance.setup_backward.get((i, j), backward_default) * b[i, j, k]
            for i in tasks
            for j in tasks
        )
        model.addConstr(processing + forward_setups + backward_setups <= cycle * u[k], name=f"cycle_{k}")

    # Ziel: Anzahl genutzter Stationen minimieren
    model.setObjective(gp.quicksum(u[k] for k in stations), GRB.MINIMIZE)

    return model


def build_and_optimize(
    instance_path: Path,
    *,
    max_stations: Optional[int] = None,
    time_limit: Optional[float] = None,
) -> gp.Model:
    """Convenience-Funktion: Instanz parsen, Modell bauen und optimieren."""

    instance = parse_alb_file(instance_path)
    model = build_sualb_model(instance, max_stations=max_stations, model_name=instance_path.name)
    if time_limit is not None:
        model.setParam("TimeLimit", time_limit)
    model.optimize()
    return model


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Solve SUALBSP-I instance with Gurobi.")
    parser.add_argument("instance", type=Path, help="Pfad zur .alb-Instanz")
    parser.add_argument("--max-stations", type=int, default=None, help="Obergrenze für Stationen (optional)")
    parser.add_argument("--time-limit", type=float, default=None, help="Gurobi-Zeitlimit in Sekunden")

    args = parser.parse_args()

    model = build_and_optimize(
        args.instance,
        max_stations=args.max_stations,
        time_limit=args.time_limit,
    )

    # Ergebnisse kurz ausgeben
    if model.status == GRB.OPTIMAL or model.status == GRB.TIME_LIMIT:
        print(f"Stations (objective): {model.objVal}")
    else:
        print(f"Optimization ended with status {model.status}")
