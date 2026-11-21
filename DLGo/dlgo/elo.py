"""Elo tracking for training runs.

Each training iteration saves a generation snapshot and plays matches against
earlier generations. All match results are appended to a CSV; ratings are
re-fitted from the full match history with a Bradley-Terry model (the
statistical model behind Elo), anchored so that generation 0 is 0 Elo.
"""
import csv
import html
import math
import os

MATCH_FIELDS = ['iteration', 'total_games', 'player', 'opponent', 'wins', 'losses']
RATING_FIELDS = ['generation', 'total_games', 'elo', 'games_played']


def append_match(path, iteration, total_games, player, opponent, wins, losses):
    new = not os.path.exists(path)
    with open(path, 'a', newline='') as f:
        w = csv.DictWriter(f, fieldnames=MATCH_FIELDS)
        if new:
            w.writeheader()
        w.writerow(dict(iteration=iteration, total_games=total_games,
                        player=player, opponent=opponent,
                        wins=wins, losses=losses))


def load_matches(path):
    if not os.path.exists(path):
        return []
    with open(path, newline='') as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k in ('iteration', 'total_games', 'wins', 'losses'):
            r[k] = int(r[k])
    return rows


def fit_ratings(matches, anchor, prior_games=1.0, iterations=2000):
    """Bradley-Terry ratings (in Elo points) from match results.

    `matches` is a list of dicts with player, opponent, wins, losses. Each
    pair that met also gets `prior_games` virtual games split evenly, which
    keeps ratings finite when one side won every game. The anchor player is
    fixed at 0 Elo. Fitted with the MM algorithm (Hunter, 2004).
    """
    wins = {}       # wins[(a, b)] = games a won against b
    players = set()
    for m in matches:
        a, b = m['player'], m['opponent']
        players.update((a, b))
        wins[(a, b)] = wins.get((a, b), 0.0) + m['wins']
        wins[(b, a)] = wins.get((b, a), 0.0) + m['losses']
    for (a, b) in list(wins):
        wins[(a, b)] += prior_games / 2.0
    if not players:
        return {}
    players = sorted(players)
    games = {}
    for (a, b), w in wins.items():
        games[(a, b)] = games.get((a, b), 0.0) + w
        games[(b, a)] = games.get((b, a), 0.0) + w
    for key in games:
        games[key] /= 2.0   # each game was counted from both sides
    total_wins = {p: sum(w for (a, _), w in wins.items() if a == p) for p in players}
    opponents = {p: [b for (a, b) in games if a == p] for p in players}

    gamma = {p: 1.0 for p in players}
    for _ in range(iterations):
        new = {}
        for p in players:
            denom = sum(games[(p, q)] / (gamma[p] + gamma[q]) for q in opponents[p])
            new[p] = total_wins[p] / denom if denom > 0 else gamma[p]
        scale = new.get(anchor, 1.0)
        new = {p: g / scale for p, g in new.items()}
        delta = max(abs(math.log(new[p] / gamma[p])) for p in players)
        gamma = new
        if delta < 1e-10:
            break
    return {p: 400.0 * math.log10(g) for p, g in gamma.items()}


def write_ratings(path, ratings, matches):
    """Write one row per generation, ordered by training games."""
    rows = rating_rows(ratings, matches)
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=RATING_FIELDS)
        w.writeheader()
        w.writerows(rows)
    return rows


def rating_rows(ratings, matches):
    """One dict per generation (RATING_FIELDS), ordered by training games."""
    first_seen = {}
    played = {}
    for m in matches:
        first_seen.setdefault(m['player'], m['total_games'])
        for p in (m['player'], m['opponent']):
            played[p] = played.get(p, 0) + m['wins'] + m['losses']
    first_seen.setdefault('gen_0000', 0)
    rows = sorted(
        (dict(generation=p, total_games=first_seen.get(p, 0),
              elo=round(e, 1), games_played=played.get(p, 0))
         for p, e in ratings.items()),
        key=lambda r: (r['total_games'], r['generation']))
    return rows


def write_html_report(path, rows, matches, title='RLGo training: Elo progress',
                      reference=None, promoted=(), gate_threshold=0.55):
    """Self-contained HTML page with an SVG Elo chart and the match table.

    reference: the generation that is currently the gating opponent (the
    best agent so far by the gating rule); promoted: the generations that
    became the reference, in order. Both are highlighted on the chart."""
    W, H, L, R, T, B = 760, 340, 60, 20, 20, 44
    xs = [r['total_games'] for r in rows] or [0]
    ys = [r['elo'] for r in rows] or [0]
    x0, x1 = min(xs), max(max(xs), 1)
    y0, y1 = min(min(ys), 0), max(max(ys), 1)
    pad = (y1 - y0) * 0.1 or 10
    y0, y1 = y0 - pad, y1 + pad

    def px(x):
        return L + (x - x0) / ((x1 - x0) or 1) * (W - L - R)

    def py(y):
        return T + (y1 - y) / ((y1 - y0) or 1) * (H - T - B)

    pts = ' '.join('%.1f,%.1f' % (px(x), py(y)) for x, y in zip(xs, ys))
    ticks = []
    for k in range(6):
        yv = y0 + (y1 - y0) * k / 5
        ticks.append('<line x1="%d" x2="%d" y1="%.1f" y2="%.1f" class="grid"/>'
                     '<text x="%d" y="%.1f" class="lab" text-anchor="end">%.0f</text>'
                     % (L, W - R, py(yv), py(yv), L - 8, py(yv) + 4, yv))
        xv = x0 + (x1 - x0) * k / 5
        ticks.append('<text x="%.1f" y="%d" class="lab" text-anchor="middle">%.0f</text>'
                     % (px(xv), H - B + 18, xv))
    promoted = [g for g in promoted if g != 'gen_0000']
    promoted_set = set(promoted)
    by_gen = {r['generation']: r for r in rows}

    def dot(r):
        g = r['generation']
        x, y = px(r['total_games']), py(r['elo'])
        tip = html.escape('%s: %.0f Elo after %d games' % (g, r['elo'], r['total_games']))
        if g == reference:
            return ('<circle cx="%.1f" cy="%.1f" r="7" class="ref"><title>%s (current '
                    'gate opponent)</title></circle><text x="%.1f" y="%.1f" class="reflab" '
                    'text-anchor="middle">gate: %s</text>'
                    % (x, y, tip, x, max(y - 12, 12), html.escape(g)))
        if g in promoted_set:
            return ('<circle cx="%.1f" cy="%.1f" r="5" class="prom"><title>%s (promoted)'
                    '</title></circle>' % (x, y, tip))
        return ('<circle cx="%.1f" cy="%.1f" r="3.5" class="dot"><title>%s</title></circle>'
                % (x, y, tip))
    dots = ''.join(dot(r) for r in rows)
    match_rows = ''.join(
        '<tr><td>%d</td><td>%d</td><td>%s</td><td>%s</td><td>%d-%d</td><td>%.0f%%</td></tr>'
        % (m['iteration'], m['total_games'], html.escape(m['player']), html.escape(m['opponent']),
           m['wins'], m['losses'], 100.0 * m['wins'] / max(1, m['wins'] + m['losses']))
        for m in reversed(matches))
    best = max(rows, key=lambda r: r['elo']) if rows else None
    summary = ('Latest: <b>%s</b> at <b>%.0f Elo</b> (gen 0 = 0) after %d self-play games. '
               'Best: %s at %.0f.' % (rows[-1]['generation'], rows[-1]['elo'], rows[-1]['total_games'],
                                     best['generation'], best['elo'])) if rows else 'No results yet.'
    ref_row = by_gen.get(reference)
    if ref_row is not None:
        gate = ('<div class="gate"><span class="k">Current best (gate opponent)</span>'
                '<span class="v">%s</span><span class="s">%.0f Elo &middot; reference since '
                '%d self-play games &middot; the learner must win at least %.0f%% of the '
                'gating games against it to replace it</span></div>'
                % (html.escape(reference), ref_row['elo'], ref_row['total_games'],
                   100 * gate_threshold))
    elif reference:
        gate = ('<div class="gate"><span class="k">Current best (gate opponent)</span>'
                '<span class="v">%s</span><span class="s">starting agent</span></div>'
                % html.escape(reference))
    else:
        gate = ''
    history = ''.join(
        '<tr%s><td>%s</td><td>%d</td><td>%.0f</td></tr>'
        % (' class="cur"' if g == reference else '', html.escape(g),
           by_gen[g]['total_games'], by_gen[g]['elo'])
        for g in reversed(promoted) if g in by_gen)
    if history:
        history = ('<h2>Promotions</h2><p>Generations that won their gating match and '
                   'became the reference, newest first.</p><div class="wrap"><table>'
                   '<tr><th>Generation</th><th>Promoted after games</th><th>Elo now</th></tr>'
                   '%s</table></div>' % history)
    legend = ('<p class="legend"><span class="sw dot"></span>generation'
              '<span class="sw prom"></span>promoted to reference'
              '<span class="sw ref"></span>current gate opponent</p>')
    page = """<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>RLGo Elo progress</title>
<style>
:root{--bg:#fafaf9;--fg:#1c1917;--muted:#78716c;--line:#e7e5e4;--accent:#2563eb;--prom:#16a34a;--gate:#d97706}
@media (prefers-color-scheme: dark){:root{--bg:#1c1917;--fg:#f5f5f4;--muted:#a8a29e;--line:#44403c;--accent:#60a5fa;--prom:#4ade80;--gate:#fbbf24}}
body{background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,sans-serif;margin:0;padding:24px 16px}
main{max-width:800px;margin:0 auto}
h1{font-size:20px;margin:0 0 4px} p{color:var(--muted);margin:0 0 16px}
svg{width:100%%;height:auto} .grid{stroke:var(--line)} .lab{fill:var(--muted);font-size:11px}
.ln{fill:none;stroke:var(--accent);stroke-width:2} .dot{fill:var(--accent)}
.prom{fill:var(--prom)} .ref{fill:var(--bg);stroke:var(--gate);stroke-width:3}
.reflab{fill:var(--gate);font-size:11px;font-weight:600}
h2{font-size:16px;margin:24px 0 4px}
.gate{display:flex;flex-wrap:wrap;align-items:baseline;gap:4px 12px;border:1px solid var(--line);
border-left:4px solid var(--gate);border-radius:6px;padding:10px 14px;margin:8px 0 16px}
.gate .k{color:var(--muted);font-size:13px} .gate .v{font-size:18px;font-weight:700}
.gate .s{color:var(--muted);font-size:13px;flex-basis:100%%}
.legend{font-size:12px;margin:0 0 4px} .sw{display:inline-block;width:10px;height:10px;
border-radius:50%%;margin:0 4px 0 14px;vertical-align:-1px} .sw:first-child{margin-left:0}
.sw.dot{background:var(--accent)} .sw.prom{background:var(--prom)}
.sw.ref{border:2px solid var(--gate);width:8px;height:8px}
tr.cur td{font-weight:700}
table{border-collapse:collapse;width:100%%;font-size:13px;margin-top:16px}
th,td{text-align:left;padding:4px 8px;border-bottom:1px solid var(--line)} th{color:var(--muted);font-weight:600}
.wrap{overflow-x:auto}
</style></head><body><main>
<h1>%s</h1>%s<p>%s Ratings are a Bradley-Terry fit over all matches below; regenerated every iteration.</p>
%s<svg viewBox="0 0 %d %d" role="img" aria-label="Elo versus self-play games">
%s<polyline points="%s" class="ln"/>%s
<text x="%d" y="%d" class="lab" text-anchor="middle">self-play games</text>
<text x="14" y="%d" class="lab" text-anchor="middle" transform="rotate(-90 14 %d)">Elo</text>
</svg>
%s<h2>Matches</h2>
<div class="wrap"><table><tr><th>Iter</th><th>Games</th><th>Player</th><th>Opponent</th><th>W-L</th><th>Win %%</th></tr>%s</table></div>
</main></body></html>""" % (html.escape(title), gate, summary, legend, W, H, ''.join(ticks), pts,
                            dots, (L + W - R) // 2, H - 6, (T + H - B) // 2, (T + H - B) // 2,
                            history, match_rows)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(page)
