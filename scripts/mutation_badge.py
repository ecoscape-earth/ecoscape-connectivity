"""Write the mutation testing badge from the results of the last `make mutation`.

The badge shows the share of the mutants that ran that the tests catch:
(killed + timeout) / (total - caught by type check - skipped). mutmut's own
`mutmut badge` counts the mutants that mypy rejects as not caught, so it
can't show this.

Run it with `make mutation-badge`, before a release.
"""

import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

BADGE_PATH = Path(__file__).parent.parent / "badges" / "mutation-score.json"


def count_statuses() -> Counter[str]:
    """Return the number of mutants with each status, from mutmut's results.

    Each line of `mutmut results --all true` is "    name: status".
    """
    results = subprocess.run(
        ["mutmut", "results", "--all", "true"],
        capture_output=True,
        text=True,
        check=True,
    )
    return Counter(
        line.rsplit(": ", 1)[1] for line in results.stdout.splitlines() if ": " in line
    )


def mutants_caught(statuses: Counter[str]) -> float:
    """Return the share of the mutants that ran that the tests catch."""
    ran = statuses.total() - statuses["caught by type check"] - statuses["skipped"]
    return (statuses["killed"] + statuses["timeout"]) / ran


def main() -> None:
    """Write the badge, or stop if mutmut has no complete results."""
    statuses = count_statuses()
    if not statuses:
        sys.exit("No mutmut results: run `make mutation` first.")
    if statuses["not checked"]:
        sys.exit(
            f"{statuses['not checked']} mutants not checked: "
            "finish `make mutation` first."
        )
    score = mutants_caught(statuses)
    BADGE_PATH.parent.mkdir(exist_ok=True)
    badge = {
        "schemaVersion": 1,
        "label": "mutants caught",
        "message": f"{round(score * 100)}%",
        "color": "blue",
    }
    BADGE_PATH.write_text(json.dumps(badge, indent=2) + "\n")
    print(f"Wrote {BADGE_PATH.name}: {badge['message']} of the mutants that ran.")


if __name__ == "__main__":
    main()
