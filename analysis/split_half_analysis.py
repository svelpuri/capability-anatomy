"""Read-only exploratory analysis; stdlib only, no model imports or inference.

Usage: python3 -B split_half_analysis.py --artifacts artifacts --output NEW_DIRECTORY
Creates a NEW output directory and refuses to overwrite a previous analysis.
Historical metrics, thresholds, partitions, and evidence are never written.
Bootstrap intervals are pointwise descriptive intervals, not acceptance tests.
"""
import argparse
import collections
import hashlib
import json
import math
from pathlib import Path
import random
import statistics


def read(path):
    return json.loads(path.read_text())


def mean(xs):
    return statistics.fmean(xs)


def rank(xs):
    order = sorted(range(len(xs)), key=xs.__getitem__)
    out = [0.0] * len(xs)
    start = 0
    while start < len(xs):
        end = start + 1
        while end < len(xs) and abs(xs[order[end]] - xs[order[start]]) < 1e-12:
            end += 1
        for i in order[start:end]:
            out[i] = (start + end - 1) / 2 + 1
        start = end
    return out


def corr(xs, ys):
    a, b = mean(xs), mean(ys)
    va = sum((x-a)**2 for x in xs)
    vb = sum((y-b)**2 for y in ys)
    return sum((x-a)*(y-b) for x, y in zip(xs, ys)) / math.sqrt(va*vb) if va*vb else None


def rho(xs, ys):
    return corr(rank(xs), rank(ys))


def quantile(xs, p):
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return None
    v = (len(xs)-1)*p
    lo, hi = math.floor(v), math.ceil(v)
    return xs[lo] + (xs[hi]-xs[lo])*(v-lo)


def distribution(xs):
    return {"median": quantile(xs, .5), "p025": quantile(xs, .025),
            "p975": quantile(xs, .975), "undefined": sum(x is None for x in xs)}


def top_ties(xs, k=5):
    threshold = sorted(xs, reverse=True)[k-1]
    return {i for i, x in enumerate(xs) if x >= threshold-1e-12}


def jaccard(a, b):
    return len(a & b) / len(a | b)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifacts', type=Path, default=Path(__file__).resolve().parents[1] / 'artifacts')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--resamples', type=int, default=1000)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    lab = args.artifacts
    result = {"analysis_status": "post_hoc_exploratory_not_a_frozen_gate",
              "resamples": args.resamples, "seed": 20260916,
              "rank_ties": "average rank; absolute tolerance 1e-12 for float residue",
              "top5": "all ties at fifth position included; set size may exceed five",
              "sources": {}, "phase5": {}}

    def source(path):
        b = path.read_bytes()
        result['sources'][path.relative_to(lab).as_posix()] = {'bytes': len(b), 'sha256': hashlib.sha256(b).hexdigest()}

    vectors = {}
    for model in ['0.6', '1.7']:
        run = lab/f'qwen3-{model}b'
        for name in ['observations.jsonl', 'report.json', 'metrics.json', 'protocol.json']:
            source(run/name)
        rows = [json.loads(line) for line in (run/'observations.jsonl').read_text().splitlines()]
        by_condition = collections.defaultdict(list)
        for row in rows:
            by_condition[row['condition']].append(row)
        conditions = sorted(c for c in by_condition if c.startswith('scan.'))
        metrics = read(run/'protocol.json')['metrics']
        scan = {}
        stats = {
            'observations': len(rows), 'status_counts': dict(collections.Counter(r['status'] for r in rows)),
            'condition_counts': {k: len(v) for k, v in sorted(by_condition.items())},
            'unique_examples_by_partition': {s: len({r['example_id'] for r in rows if r['partition']==s}) for s in ['discovery','validation']},
            'classification_counts': dict(collections.Counter(r['classification'] for r in read(run/'report.json')['candidate_ranking'])),
            'candidates': read(run/'report.json')['candidates'],
            'repeated_rows': {}, 'baseline_metrics': {}, 'metric_analysis': {},
        }
        pairs = collections.defaultdict(list)
        for row in rows:
            pairs[(row['condition'], row['example_id'])].append(row)
        stats['repeated_rows'] = {
            'pairs': len(pairs),
            'nonidentical_output_hash_pairs': sum(len({r['output_sha256'] for r in v})>1 for v in pairs.values()),
            'nonidentical_score_pairs': sum(len({json.dumps(r['scores'],sort_keys=True) for r in v})>1 for v in pairs.values()),
        }
        control = next(c for c in by_condition if c.startswith('control.random.'))
        reference = 'scan.' + by_condition[control][0]['component_id']
        ix = lambda rs: {(r['example_id'],r['repetition']):(r['output_sha256'], r['scores']) for r in rs}
        stats['random_duplicate'] = {'condition': control, 'scan_condition': reference,
                                     'rows': len(by_condition[control]), 'hashes_and_scores_equal': ix(by_condition[control])==ix(by_condition[reference])}
        toolrows = [r for r in rows if 'full_call' in r['scores']]
        stats['binding_full_call'] = {'paired_rows': len(toolrows), 'unequal': sum(r['scores']['argument_binding']!=r['scores']['full_call'] for r in toolrows)}
        base_ix = ix(by_condition['baseline.beginning'])
        stats['baseline_controls_equal'] = {c: ix(by_condition[c])==base_ix for c in ['baseline.middle','baseline.end','control.no-op']}
        for metric in metrics:
            baseline = {r['example_id']: r['scores'][metric]['value'] for r in by_condition['baseline.beginning'] if metric in r['scores']}
            validation = {r['example_id']: r['scores'][metric]['value'] for r in by_condition['validation.baseline'] if metric in r['scores']}
            selected = [r for r in by_condition['baseline.beginning'] if metric in r['scores'] and r['repetition']==0]
            ids = sorted(baseline)
            groups = collections.defaultdict(list)
            for row in selected:
                # Duplicate instruction prompts must not become independent units.
                key = row['prompt_sha256'] if metric=='instruction_format' else row['group_id']
                groups[key].append(row['example_id'])
            group_ids = sorted(groups)
            n = len(group_ids)
            cond_values = [{r['example_id']: r['scores'][metric]['value'] for r in by_condition[c] if metric in r['scores']} for c in conditions]
            def damage(values, sample):
                # Cluster bootstrap preserves original per-record weighting.
                base = mean([baseline[i] for g in sample for i in groups[g]])
                intervention = mean([values[i] for g in sample for i in groups[g]])
                return (intervention-base)/base if metric=='perplexity' else base-intervention
            damages = [damage(v, group_ids) for v in cond_values]
            scan[metric] = damages
            stats['baseline_metrics'][metric] = {'discovery': mean(baseline.values()), 'validation': mean(validation.values()),
                'absolute_gap': abs(mean(baseline.values())-mean(validation.values())),
                'relative_gap': abs(mean(baseline.values())-mean(validation.values()))/mean(baseline.values()) if metric=='perplexity' else None,
                'unique_records_per_split': [len(baseline),len(validation)], 'independence_groups_used': n,
                'unique_discovery_prompt_hashes': len({r['prompt_sha256'] for r in selected}),
                'score_grid': 1/len(baseline) if metric!='perplexity' else None}
            rng = random.Random(20260916)
            boots = [[] for _ in conditions]
            rankboots = [[] for _ in conditions]
            topcounts = [0]*len(conditions)
            splitrho = []
            for _ in range(args.resamples):
                sample = rng.choices(group_ids, k=n)
                dv = [damage(v, sample) for v in cond_values]
                ranks = rank([-v for v in dv])
                top = top_ties(dv)
                for i,x in enumerate(dv):
                    boots[i].append(x);rankboots[i].append(ranks[i]);topcounts[i] += i in top
                shuffled = rng.sample(group_ids,n)
                half = n//2
                left = [damage(v,shuffled[:half]) for v in cond_values]
                right = [damage(v,shuffled[half:]) for v in cond_values]
                splitrho.append(rho(left,right))
            stats['metric_analysis'][metric] = {
                'split_half_rank_correlation_distribution': distribution(splitrho),
                'layers': [{'layer': int(c.rsplit('.',1)[1]),'damage': damages[i],
                            'pointwise_bootstrap_95': [quantile(boots[i],.025),quantile(boots[i],.975)],
                            'bootstrap_rank_95': [quantile(rankboots[i],.025),quantile(rankboots[i],.975)],
                            'top5_tie_inclusive_fraction': topcounts[i]/args.resamples}
                           for i,c in enumerate(conditions)],
            }
        stats['rank_correlations_between_metrics'] = {
            f'{a} vs {b}': rho(scan[a],scan[b]) for a,b in [
                ('tool_selection','argument_binding'),('tool_selection','abstention'),
                ('tool_selection','perplexity'),('tool_selection','instruction_format')]}
        result['phase5'][model] = stats
        vectors[model] = scan
    result['phase5_cross_size_rank_correlations_same_28_indices'] = {
        k: rho(vectors['0.6'][k],vectors['1.7'][k]) for k in vectors['0.6']}
    (args.output/'statistics.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'phase5':{k:{'rows':v['observations'],'classes':v['classification_counts'],
        'tool_half_rho':v['metric_analysis']['tool_selection']['split_half_rank_correlation_distribution']} for k,v in result['phase5'].items()},
        'cross_size':result['phase5_cross_size_rank_correlations_same_28_indices']},indent=2))


if __name__ == '__main__':
    main()
