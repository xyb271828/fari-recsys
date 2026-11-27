"""Render comparable tables and interpretation directly from completed JSON runs."""
import argparse
import json
from pathlib import Path

FAMILIES = ('popularity', 'time_decayed_popularity', 'ease_r', 'als', 'two_tower', 'two_tower_logq')


def point(model, split, metric):
    value = model['evaluation'][split][metric]
    return value['mean'] if isinstance(value, dict) else value


def render(results):
    lines = ['# Real-data recommendation results', '',
             'All results below use real public interactions. See [protocol](README.md) for',
             'split rules, eligibility, user aggregation, tuning budgets and limitations.', '',
             'Stochastic model entries are means across initialization seeds 42/43/44;',
             'they are not ensemble results. Recall and NDCG use equal user weights.', '']
    for title, payload in results.items():
        lines += [f'## {title}', '',
                  f"Validation-selected strongest baseline: **{payload['strongest_baseline_selected_on_validation']}**.", '',
                  f"Training: {payload['data']['training_impressions']:,} observed interactions; "
                  f"{payload['data']['training_positive_events']:,} positive events; "
                  f"{payload['data']['training_positive_items']:,} eligible items.", '']
        for split in ('test', 'random_test'):
            if split not in payload['models']['popularity']['evaluation']:
                continue
            lines += [f'### {split}', '',
                      '| Model | Recall@100 | NDCG@10 | Tail Recall@100 | Coverage@100 |',
                      '| --- | ---: | ---: | ---: | ---: |']
            for name in FAMILIES:
                model = payload['models'][name]
                values = [point(model,split,key) for key in ('recall@100','ndcg@10','tail_recall@100','coverage@100')]
                lines.append(f"| {name} | {values[0]:.4f} | {values[1]:.4f} | {values[2]:.4f} | {values[3]:.4f} |")
            lines += ['', '| Comparison | Recall delta (percentage points) | User-bootstrap 95% interval |',
                      '| --- | ---: | ---: |']
            comparisons = [('EASE vs popularity', payload['paired_user_bootstrap'][split]['ease_r']['vs_popularity']['recall@100']),
                           ('logQ tower vs raw tower', payload['logq_ablation_paired'][split]['recall@100'])]
            comparisons += [(name + ' vs validation-selected baseline',
                             payload['paired_user_bootstrap'][split][name]['vs_validation_selected_baseline']['recall@100'])
                            for name in ('two_tower','two_tower_logq')]
            for label, entry in comparisons:
                low, high = entry['ci95']
                lines.append(f"| {label} | {100*entry['delta']:+.2f} | [{100*low:+.2f}, {100*high:+.2f}] |")
            lines += ['', '| Model | Warm-user Recall@100 | Cold-user Recall@100 |', '| --- | ---: | ---: |']
            reference = payload['models']['popularity']['evaluation'][split]
            lines += [f"| popularity | {reference['warm']['recall@100']:.4f} | {reference['cold']['recall@100']:.4f} |"]
            ease = payload['models']['ease_r']['evaluation'][split]
            lines += [f"| ease_r | {ease['warm']['recall@100']:.4f} | {ease['cold']['recall@100']:.4f} |"]
            for name in ('als','two_tower','two_tower_logq'):
                models = [payload['models'][f'{name}_seed{s}']['evaluation'][split] for s in payload['seeds']]
                warm = sum(model['warm']['recall@100'] for model in models) / len(models)
                cold = sum(model['cold']['recall@100'] for model in models) / len(models)
                lines.append(f'| {name} | {warm:.4f} | {cold:.4f} |')
            lines += ['', f"Evaluation: {reference['users']:,} users "
                      f"({reference['warm']['users']:,} warm / {reference['cold']['users']:,} cold), "
                      f"{reference['requests']:,} eligible positive requests.", '']
        lines += ['### Selected checkpoints', '', '| Seed | Raw tower epoch | logQ tower epoch |', '| --- | ---: | ---: |']
        for seed in payload['seeds']:
            raw = payload['models'][f'two_tower_seed{seed}']['selection']['selected_epoch']
            logq = payload['models'][f'two_tower_logq_seed{seed}']['selection']['selected_epoch']
            lines.append(f'| {seed} | {raw} | {logq} |')
        lines += ['', f"Recorded wall time: {payload['wall_seconds']:.1f} seconds on the local CPU run.",
                  'See JSON for every seed, tuning trial, split audit and package version.', '']
    lines += ['## Interpretation boundaries', '',
              '- Only claim superiority for the named model, baseline, dataset, metric and split.',
              '- A logQ improvement over the raw tower is not evidence of improvement over ALS/EASE or popularity.',
              '- KuaiRand standard-policy and random-exposure retrieval are separate evaluations.',
              '- Random-test retrieval conditions on positive requests in the retained candidate pool; it is not CTR or causal lift.',
              '- Temporal changes, cold-user rates and candidate restrictions affect absolute values.',
              '- One split per dataset and three initialization seeds do not establish production generalization.', '']
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=Path('runs/real'))
    parser.add_argument('--output', type=Path, default=Path('reports/real/SUMMARY.md'))
    args = parser.parse_args()
    results = {name: json.loads((args.input/name/'results.json').read_text()) for name in ('movielens','kuairand')}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render(results))


if __name__ == '__main__':
    main()
