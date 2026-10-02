"""Audit existing 1x1 IGDs without producing spatial maps or training data.

Build helper: cargo build --manifest-path src/igd_extract/Cargo.toml
  --bin analyze_dictionaries --target-dir src/gdcompress/target --release --offline --locked
Run: python src/analyze_cumulative_dataset.py --output-dir output/cumulative_audit
Outputs are created exclusively, never overwritten. No third-party Python dependency.
"""
import argparse
import json
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', type=Path, default=ROOT/'data/cifar-10_compressed_1x1')
    parser.add_argument('--original-dir', type=Path, default=ROOT/'data/cifar-10_resized')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--helper', type=Path, default=ROOT/'src/gdcompress/target/release/analyze_dictionaries.exe')
    args = parser.parse_args()
    paths = sorted(args.input_dir.glob('*.igd'))
    if not paths:
        parser.error('No IGD files found')
    if not args.helper.is_file():
        parser.error('Build the Rust dictionary helper first')
    originals = {p.stem for p in args.original_dir.glob('*.png')}
    compressed = {p.stem for p in paths}
    args.output_dir.mkdir(parents=True, exist_ok=False)
    rows = []
    errors = []
    start = time.monotonic()
    with (args.output_dir/'per_image.jsonl').open('x', encoding='utf-8') as output:
        with subprocess.Popen([str(args.helper)], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                              text=True, encoding='utf-8') as process:
            for i, path in enumerate(paths, 1):
                process.stdin.write(str(path.resolve())+'\n')
                process.stdin.flush()
                line = process.stdout.readline()
                if not line:
                    raise RuntimeError('Helper terminated early; partial report retained')
                row = json.loads(line)
                output.write(json.dumps(row)+'\n')
                (errors if 'error' in row else rows).append(row)
                if i % 500 == 0 or i == len(paths):
                    print(f'{i}/{len(paths)}; errors={len(errors)}; {time.monotonic()-start:.1f}s', flush=True)
            process.stdin.close()
            if process.wait() != 0:
                raise RuntimeError('Helper failed')
    if not rows:
        raise RuntimeError('No valid images; see per_image.jsonl')
    counts = sorted(r['num_bases'] for r in rows)
    def quantile(p):
        return counts[round((len(counts)-1)*p)]
    by_s = []
    for s in range(17):
        worst = max(rows, key=lambda r:r['zmax_s0_to_16'][s])
        by_s.append(dict(s=s, max_z=worst['zmax_s0_to_16'][s], worst_image=worst['path'],
                         overflow_uint8=sum(r['zmax_s0_to_16'][s]>255 for r in rows),
                         overflow_uint16=sum(r['zmax_s0_to_16'][s]>65535 for r in rows)))
    limits = {}
    for dtype in ('uint8','uint16'):
        finite = [r for r in rows if not r['s_unbounded']]
        impossible = [r for r in finite if r['max_s_'+dtype] is None]
        limiting = min(finite, key=lambda r:r['max_s_'+dtype] if r['max_s_'+dtype] is not None else -1) if finite else None
        limits[dtype] = dict(max_common_s=None if impossible or limiting is None else limiting['max_s_'+dtype],
                            impossible_even_at_s0=len(impossible),
                            limiting_image=limiting['path'] if limiting else None,
                            unbounded=not finite)
    report = dict(schema='cumulative-dictionary-audit-v1', input_dir=str(args.input_dir.resolve()),
        original_dir=str(args.original_dir.resolve()), original_count=len(originals),
        igd_count=len(paths), valid_count=len(rows), errors=errors,
        missing_igd_count=len(originals-compressed), extra_igd_count=len(compressed-originals),
        complete_for_originals=bool(originals) and not (originals-compressed) and not errors,
        conventions=dict(base_order='Selected chunk positions ascending; first bit treated as MSB, matching network features',
            formula='q=round_half_up(S*log2(1+delta)/log2(1+max_delta)); zmax=K-1+sum(q)',
            scope='Dictionary only: does not verify spatial sample stream or reconstruct RGB',
            precision='float64 normalization; integer cumulative sum; grouping 1x1, at most 32 base bits'),
        bases=dict(min=counts[0], median=quantile(.5), p95=quantile(.95), p99=quantile(.99), max=counts[-1],
                   worst_images=[r['path'] for r in rows if r['num_bases']==counts[-1]]),
        all_gaps_equal_count=sum(r['all_gaps_equal'] for r in rows),
        limits=limits, by_s=by_s, elapsed_seconds=time.monotonic()-start)
    (args.output_dir/'summary.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    (args.output_dir/'missing_igd.txt').write_text('\n'.join(sorted(originals-compressed)),encoding='utf-8')
    print(json.dumps({'bases':report['bases'],'limits':limits,'valid_count':len(rows),'errors':len(errors)},indent=2))


if __name__ == '__main__':
    main()
