import os
import sys
from pathlib import Path
project_root = Path(__file__).resolve().parent
sys.path.append(str(project_root))

from rl.evaluate_and_visualize import visualize_run

import matplotlib
matplotlib.use('Agg') 

ALGO = "best_incremental_eval"
SCENARIOS = [
    "butterfly_scA",
    "butterfly_scB",
    "butterfly_scC",
    "butterfly_scD",
    "butterfly_scE",
]

for sc in SCENARIOS:
    print(f"\nVisualizing {sc} (algorithm: {ALGO})...")
    visualize_run(
        dataset=sc,
        algorithm=ALGO,
        run_id=1,  # use valid run_id since data should actually be generated for run 1 initially
        variable='density',
        vis_actions=True,
        save_gif=True # this will now save as mp4
    )

print("Done visualizing!")
