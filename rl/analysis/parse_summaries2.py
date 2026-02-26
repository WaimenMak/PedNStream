import json
import glob
import numpy as np
from datetime import datetime

summaries = glob.glob('checkpoints/experiments/incremental_*/experiment_summary.json')
results = []
for s in summaries:
    with open(s, 'r') as f:
        data = json.load(f)
        evals = data.get('eval_results', {})
        # check if it's the new run by looking at the timestamp (after 2026-02-21 12:00:00)
        ts = data.get('timestamp')
        if ts:
            try:
                dt = datetime.fromisoformat(ts)
                if dt < datetime(2026, 2, 21, 12, 0): # New runs started after 12:00
                    continue
            except:
                pass
        if evals:
            avg_eval_scenarios = ['butterfly_scA', 'butterfly_scB', 'butterfly_scC', 'butterfly_scD', 'butterfly_scE']
            valid_evals = [evals[k]['avg_reward'] for k in avg_eval_scenarios if k in evals]
            if valid_evals:
                avg_eval = np.mean(valid_evals)
                results.append((data['experiment_tag'], avg_eval, evals))

results.sort(key=lambda x: x[1], reverse=True)
for tag, avg_eval, evals in results:
    print(f"{tag}:")
    print(f"  A-E AVERAGE: {avg_eval:.2f}")
    for sc, sc_data in evals.items():
        print(f"  {sc}: {sc_data['avg_reward']:.2f}")
    print()
