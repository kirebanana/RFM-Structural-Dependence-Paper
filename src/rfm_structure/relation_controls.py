"""Scientifically defined FK groups for paired within-context controls."""


def relation_groups(registry):
    """Separate task-row links and ordinary database incidence by schema meaning.

    Groups overlap deliberately: database_all is the aggregate control, while
    its driver/constructor/race/circuit subsets identify different mechanisms.
    Sparse individual columns are pooled by parent type; counts remain explicit.
    """
    database_tables = {
        "results", "qualifying", "standings", "constructor_results",
        "constructor_standings", "races", "drivers", "constructors", "circuits",
    }
    predicates = {
        "task_driver": lambda r: r["child_table"] == "driver-dnf" and r["parent_table"] == "drivers",
        "database_all": lambda r: r["child_table"] in database_tables,
        "database_driver": lambda r: r["child_table"] in database_tables and r["parent_table"] == "drivers",
        "database_constructor": lambda r: r["child_table"] in database_tables and r["parent_table"] == "constructors",
        "database_race": lambda r: r["child_table"] in database_tables and r["parent_table"] == "races",
        "race_circuit": lambda r: r["child_table"] == "races" and r["parent_table"] == "circuits",
    }
    groups = {name: sorted(int(key) for key, info in registry.items() if predicate(info))
              for name, predicate in predicates.items()}
    if any(not ids for ids in groups.values()):
        raise ValueError("rel-f1 metadata lacks one of the declared relation groups")
    if set(groups["task_driver"]) & set(groups["database_all"]):
        raise ValueError("task and database relation controls must be disjoint")
    return groups
