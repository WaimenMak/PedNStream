import os
import sys
from pathlib import Path
project_root = Path(__file__).resolve().parent
sys.path.append(str(project_root))

from rl.evaluate_and_visualize import visualize_run

import matplotlib
matplotlib.use('macosx') 

ALGO = "best_incremental_eval"
SCENARIOS = [
    "butterfly_scB"
]

for sc in SCENARIOS:
    print(f"\nVisualizing {sc} (algorithm: {ALGO})...")
    visualize_run(
        dataset=sc,
        algorithm=ALGO,
        run_id=None,
        variable='density',
        vis_actions=True,
        save_gif=True
    )

print("Done generating all GIFs!")
