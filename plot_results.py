"""Render measured CSV data as standalone SVG without extra dependencies."""
import argparse
import csv
import html
from pathlib import Path


def text(x, y, value, size=13, anchor='start', color='#334155'):
    return f'<text x="{x}" y="{y}" font-size="{size}" text-anchor="{anchor}" fill="{color}">{html.escape(str(value))}</text>'


def start(title, subtitle, height=700):
    return [f'<svg xmlns="http://www.w3.org/2000/svg" width="1100" height="{height}" viewBox="0 0 1100 {height}">', '<rect width="100%" height="100%" fill="white"/>', '<g font-family="Arial, sans-serif">', text(40, 36, title, 23), text(40, 62, subtitle, 13)]


def panel(parts, top, title, series, ymin, ymax, xmax):
    left, width, height = 85, 940, 190
    parts.append(text(left, top-15, title, 16))
    def point(x, y):
        return left + width*x/xmax, top+height-height*(y-ymin)/(ymax-ymin)
    for i in range(6):
        value = ymin + (ymax-ymin)*i/5
        _, y = point(0, value)
        parts.append(f'<path d="M {left} {y} H {left+width}" stroke="#e2e8f0"/>')
        parts.append(text(left-12, y+4, f'{value:.1f}', anchor='end'))
        x, _ = point(xmax*i/5, 0)
        parts.append(text(x, top+height+22, f'{xmax*i/5:,.0f}', anchor='middle'))
    for label, color, data in series:
        points = ' '.join(f'{point(x,y)[0]:.2f},{point(x,y)[1]:.2f}' for x,y in data)
        parts.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2"/>')
    parts.append(text(left+width/2, top+height+45, 'Training episodes', anchor='middle'))
    x = left
    for label, color, _ in series:
        parts.append(text(x, top+height+70, label, 12, color=color))
        x += 330


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('run_dir')
    root = Path(parser.parse_args().run_dir)
    comparison = list(csv.DictReader((root/'comparison.csv').open()))
    curve = list(csv.DictReader((root/'learning_curve.csv').open()))
    training = [r for r in curve if r['agent']=='abstract_q']
    evaluation = [r for r in comparison if r['agent']=='abstract_q']
    means = []
    for i in range(0, len(training), 100):
        batch = training[i:i+100]
        means.append((int(batch[-1]['episode']), sum(float(r['score']) for r in batch)/len(batch)))
    xmax = int(training[-1]['episode'])
    score = [(int(r['training_episodes']), float(r['avg_score'])) for r in evaluation]
    parts = start('Q-learning learning curve', 'Training seed 42; frozen evaluation seeds 10000–10099; one deterministic environment', 760)
    panel(parts, 110, 'Episode return', [('Training: 100-episode mean','#2563eb',means), ('Frozen evaluation mean','#d97706',score), ('Rule baseline = 6.63','#15803d',[(0,6.63),(xmax,6.63)])], -5, 8, xmax)
    series = [(label,color,[(int(r['training_episodes']),float(r[key])) for r in evaluation]) for label,color,key in [('Trays served','#2563eb','avg_served'),('Items burnt','#dc2626','avg_burnt')]]
    panel(parts, 440, 'Evaluation outcomes per episode', series, 0, 25, xmax)
    parts.append('</g></svg>')
    (root/'learning_curve.svg').write_text('\n'.join(parts))
    selected = [r for r in comparison if r['agent'] in ('random_all','random_valid','random_abstract','rule') or (r['agent']=='raw_q' and r['training_episodes']=='1000') or (r['agent']=='abstract_q' and r['training_episodes'] in ('1000','10000','50000'))]
    parts = start('Agent comparison', 'Mean of 100 evaluation episodes, 200 steps each. Score = cumulative reward; served = 8-item trays.', 680)
    for col, (metric, title, lo, hi) in enumerate([('avg_score','Return',-12,8),('avg_served','Trays served',0,5),('avg_burnt','Items burnt',0,45)]):
        left = 260 + col*275
        width = 215
        parts.append(text(left, 107, title, 16))
        zero = left+width*(0-lo)/(hi-lo)
        parts.append(f'<path d="M {zero} 120 V 610" stroke="#cbd5e1"/>')
        for i,r in enumerate(selected):
            y=140+i*57
            if col == 0:
                label=r['agent'] + (' '+r['training_episodes'] if r['training_episodes']!='0' else '')
                parts.append(text(30,y+19,label,13))
            value=float(r[metric]); end=left+width*(value-lo)/(hi-lo)
            color='#15803d' if r['agent']=='rule' else '#2563eb' if r['agent']=='abstract_q' else '#94a3b8'
            parts.append(f'<rect x="{min(zero,end)}" y="{y}" width="{max(.5,abs(end-zero))}" height="28" fill="{color}"/>')
            parts.append(text(left+width+5,y+19,f'{value:.2f}',12))
    parts.append('</g></svg>')
    (root/'agent_comparison.svg').write_text('\n'.join(parts))


if __name__ == '__main__':
    main()
