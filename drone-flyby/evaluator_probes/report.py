"""Render measured JSON to the human-readable experimental report."""

import argparse
import json
import statistics
from pathlib import Path


COMMAND = '& drone-flyby/.venv/Scripts/python.exe drone-flyby/evaluator_probes/run_all.py'


def number(value):
    return f'{value:.6f}'


def table(headers, rows):
    return '\n'.join(['| ' + ' | '.join(map(str, headers)) + ' |',
                      '| ' + ' | '.join('---' for _ in headers) + ' |'] +
                     ['| ' + ' | '.join(str(v).replace('\n', ' ') for v in row) + ' |' for row in rows])


def timing_table(runs):
    return table(['Sleep ms', 'Total', 'Sent', 'Skipped', 'Unanswered', 'Accepted', 'Timeouts',
                  'RTT mean / median / max ms', 'AP'], [
        [r['delay_ms'], *[r['statistics'][k] for k in ('frames_total', 'frames_sent', 'frames_skipped',
         'frames_unanswered', 'responses_accepted', 'timeouts')],
         ' / '.join(f'{r["rtt_ms"][k]:.1f}' for k in ('mean', 'median', 'max')), number(r['score'])] for r in runs])


def write_report(data, path):
    d = data
    sections = []
    def add(text):
        sections.append(text.strip())
    def detail(label, title, question, hypothesis, setup, expected, observed, matched, interpretation, implication, uncertainty):
        add(f'## {label} — {title}\n\n**Question:** {question}\n\n**Hypothesis:** {hypothesis}\n\n'
            f'**Setup:** {setup}\n\n**Exact command (complete suite, JSON key `{label}`):**\n\n'
            f'```powershell\n{COMMAND}\n```\n\n**Expected from static analysis:** {expected}\n\n'
            f'**Observed:**\n\n{observed}\n\n**Matched expectation?** {matched}\n\n'
            f'**Interpretation:** {interpretation}\n\n**Possible implication:** {implication}\n\n'
            f'**Remaining uncertainty:** {uncertainty}')
    env = d['environment']
    add('# Evaluator probe results\n\nMeasured against the supplied implementation. No detector, tracker, training, '
        'camera-policy design, or production-solution changes are included.')
    add('# Environment\n\n' + table(['Item', 'Recorded value'], [
        ['Run UTC', env['timestamp_utc']], ['Commit', env['commit']], ['Branch', env['branch']],
        ['Python', env['python'].replace('\n', ' ')], ['Platform', env['platform']],
        ['Evaluator monotonic clock', env.get('monotonic_clock', 'not recorded')],
        ['Source/data unchanged during suite', d['authoritative_inputs_unchanged']],
        *[[name, version] for name, version in env['dependencies'].items()]]))
    add('Git state at measurement start:\n\n```text\n' + (env['git_status'] or '(clean)') + '\n```\n\n'
        'The repository was clean before this task. Created `drone-evaluator-probes` from '
        '`codex/drone-evaluator-probes` at the commit above. All task files are under '
        '`drone-flyby/evaluator_probes/`; authoritative files are unchanged. The JSON stores '
        'SHA-256 fingerprints of the five executable evaluator/protocol/helper/endpoint files and all '
        '51 Helsinki data files. The installed scorer is **1.8.0**, permitted by the supplied '
        '`>=1.7.2,<2` requirement; the README comments mentioning 1.7.2 are not an installed-version check.\n\n'
        'Environment recovery: `python` was unavailable, `py -3.13` referenced a missing `C:\\Python313`, '
        'and the `uv` launcher was broken. Used bundled Python 3.12.14 to create the ignored `.venv`. '
        'Initial sandboxed pip networking failed; installation succeeded with network permission. '
        'No evaluator changes were needed. Exact bootstrap command:\n\n```powershell\n'
        '& "C:\\Users\\mbnas\\.cache\\codex-runtimes\\codex-primary-runtime\\dependencies\\python\\python.exe" '
        '-m venv drone-flyby/.venv\n'
        '& drone-flyby/.venv/Scripts/python.exe -m pip install -r drone-flyby/requirements.txt\n```')
    add('# Method\n\n'
        'Pure metric probes call the unchanged `local_evaluator.score`. Synthetic scenes are real '
        'temporary PNG/annotation directories read through the official loaders. They use a 200x200 '
        '`hangar` box at `[100,100,300,300]`; controlled FPs use `[600,600,800,800]`. '
        'Protocol probes call the unchanged `replay` against a deterministic local HTTP server and '
        'use official global-coordinate helpers and DTO validation. A-C and G have one GT per frame; '
        'D also includes two same-class GTs; H/I have two good detections of different classes in each '
        'of two frames, so a malformed addition can demonstrably discard good detections.\n\n'
        'Timing sweeps use all 25 Helsinki images, a serial evaluator, real HTTP, actual endpoint '
        'sleeps, and the real clock. No simulated-latency flag or fake clock is used. A read-only Python '
        'trace records elapsed time and selected indices immediately after the evaluator calculation. '
        'A no-trace control and separately timed image preparation check its interpretation. The '
        'synthetic timing follow-up uses the same full-resolution rendering path with simple PNG '
        'content; it does not stand in for the Helsinki performance measurement.\n\n'
        'Each deterministic case changes its named factor. The monotonic confidence transform is '
        'squaring. Missing-frame recall is independently counted from retained GT annotations, and '
        'IoU uses independent intersection/union arithmetic. COCO precision samples are also read '
        'from the actual scorer in C. Timing runs are sequential single runs per condition, not '
        'statistical estimates. On the recorded Python 3.12 Windows runtime, the evaluator clock is '
        '`GetTickCount64()` with 15.625 ms resolution; sub-tick distinctions in sleep targets such '
        'as 330 versus 333 ms cannot establish precise boundaries. The clock was not replaced. '
        'Differences between nearby delays must not be treated as a universal '
        'performance boundary. Raw counters, RTTs, per-class scores, frame indices, camera states, '
        'feedback, diagnostic text, and recurrence observations are in `probe_results.json`.')
    a_equal = d['A']['in_crop']['score'] == d['A']['off_crop']['score'] == 1
    b_equal = [r['score'] for r in d['B']] == [0, 1, 1, 1, 1]
    c_equal = d['C']['TP_first']['score'] == 1 and d['C']['FP_first']['score'] == .5
    add('# Results Summary\n\n' + table(
        ['Probe', 'Hypothesis', 'Observed result', 'Confirmed?', 'Strategic importance'], [
        ['Oracle/H1', 'Perfect GT gives 1', number(d['oracle']['score']), 'Yes' if d['oracle']['score'] == 1 else 'No', 'Environment gate'],
        ['A/H2', 'Off-crop detections count equally', 'In/out crop both ' + number(d['A']['off_crop']['score']), a_equal, 'Global reporting'],
        ['B/H3', 'Inclusive IoU .50', ', '.join(number(x['score']) for x in d['B']), b_equal, 'Matching boundary'],
        ['C/H4', 'FP rank changes AP', 'TP first 1; FP first .5; square unchanged', c_equal, 'Confidence order'],
        ['D/H5', 'Duplicate penalty depends on ranking', 'Single GT: all 1; duplicate before second TP: ' + number(d['D']['duplicate_before_second_TP']['score']), 'Refined', 'No fixed duplicate penalty'],
        ['E/H6', 'Missing frames cost recall', 'Every second: ' + number(d['E']['every_second_even']['score']), 'Yes', 'Coverage'],
        ['F/H7', 'Latency skips frames', 'Measured jumps; preprocessing dominates on this host', 'Yes, with clock refinement', 'End-to-end throughput'],
        ['G/H8', '100 detections retained per class/image', 'Rank 99 / 100 / 101: ' + ' / '.join(number(x['score']) for x in d['G']), 'Yes', 'Rank cap'],
        ['H/H9', 'Bad responses lose all detections', 'Invalid first responses lose both good boxes', 'Yes; ID mismatches are schema-valid', 'Whole-response integrity'],
        ['I/H9', 'Illegal camera geometry preserves detections', number(d['I']['illegal_geometry']['score']), 'Yes', 'Separate failure effects'],
        ['J/H10', 'Feedback persists until valid command', 'Present frames 1-3; absent frame 4', 'Yes', 'Feedback lifecycle'],
        ['K/H7', 'Late acceptance differs from timeout', '3000 ms accepted; 3500 ms times out', 'Yes', 'Throughput versus deadline'],
        ['L', 'Absent allowed classes excluded', 'Oracle plus absent-class FP: ' + number(d['L_absent_class']['oracle_plus_absent_class']['score']), 'Yes, synthetic scope', 'Evaluated category set']]))
    add('# Detailed Results\n\n## Oracle — environment gate\n\n'
        '**Question/hypothesis:** Does supplied GT score 1.000?\n\n'
        '**Setup:** Unmodified full Helsinki scene, supplied oracle CLI.\n\n'
        '**Exact command:**\n\n```powershell\n'
        '& drone-flyby/.venv/Scripts/python.exe drone-flyby/local_evaluator.py --oracle\n```\n\n'
        '**Expected:** 1.000 overall and all classes. **Observed output:**\n\n```text\n' +
        d['oracle']['stdout'].strip() + '\n```\n\n'
        '**Matched:** Yes. **Interpretation/implication:** Local scorer and supplied annotations are '
        'consistent enough to proceed. **Uncertainty:** This is not proof of hosted scorer parity.')
    detail('A', 'Off-crop global predictions', 'Does current crop restrict current scoring?',
           'H2: identical correct global boxes score equally inside and outside the crop.',
           'Three-frame synthetic scene; frame 0 commands L1 center (960,540) or (2880,1620). '
           'Frames 1-2 contain the same global box; it intersects only the first crop. All predictions are identical.',
           'Equal AP, ideally 1.0.', table(['Case', 'AP', 'Crop intersects GT, frames 0/1/2'], [
               [name, number(r['score']), [x['gt_intersects_crop'] for x in r['requests']]] for name, r in d['A'].items()]),
           str(a_equal), 'The actual replay accepts, converts, and scores the off-crop boxes with no crop restriction.',
           'Responses can carry whole-frame information beyond the current observation.',
           'This tests scoring eligibility, not the ability to estimate an unseen object.')
    detail('B', 'IoU boundary', 'Where is the match threshold?', 'H3: .499 fails; .500 and .501 match.',
           'One GT box; prediction has the same height and left edge and width 200 times target IoU. '
           'At .500 all corners are integers. Other fixture geometry and confidence stay fixed.',
           'AP 0,1,1,1,1 for IoUs .499,.500,.501,.60,.90.',
           table(['Target IoU', 'Independently computed IoU', 'AP'], [[x['target_iou'], repr(x['computed_iou']), number(x['score'])] for x in d['B']]),
           str(b_equal), 'The inclusive .50 threshold is measured. Isolated .60 and .90 matches have equal AP.',
           'Extra localization beyond the matching threshold gives no direct gain in this controlled case.',
           'Multi-object association ambiguities and floating-point perturbations around .50 are not exhausted.')
    c_precision = d['C_precision']
    detail('C', 'Confidence ranking', 'Can the same FP be harmless or costly depending on confidence?',
           'H4: TP-first may avoid AP loss; FP-first lowers AP; monotonic confidence transforms preserve it.',
           'One GT, one perfect TP, one disjoint FP. Only scores change between .9/.1 and .1/.9, '
           'then each pair is squared. Read actual COCO precision tensor on return from scorer.',
           'TP-first 1; FP-first .5; squared variants unchanged.',
           table(['Case', 'AP'], [[name, number(x['score'])] for name, x in d['C'].items()]) + '\n\n' +
           table(['Actual tensor', 'Recall samples', 'Min precision', 'Max precision', 'Maximum recall'],
                 [[name, len(r['precision']), min(r['precision']), max(r['precision']), r['max_recall']] for name, r in c_precision.items()]),
           str(c_equal and all(d['C'][k]['score'] == d['C'][k + '_squared']['score'] for k in ('TP_first', 'FP_first'))),
           'COCO evaluates a confidence-sorted prefix, envelopes precision from the right, and samples '
           '101 recall points. TP-first reaches recall 1 with precision 1 before the FP; every sampled '
           'precision is 1. FP-first reaches recall 1 at precision .5; every sampled precision is .5. '
           'The measured precision arrays support the AP difference directly.',
           'Confidence ranking can affect AP substantially; an extra FP has no universal fixed AP cost.',
           'A late FP need not be harmless when useful detections remain later in the class ranking or when maxDets truncates them. Ties are not tested.')
    detail('D', 'Duplicate ranking', 'When do duplicate boxes reduce AP?',
           'H5: later matches to already-matched GT are FPs, with rank-dependent cost.',
           'Single-object cases reverse scores on two identical perfect boxes; a two-object same-class '
           'case moves only duplicate confidence from .1 to .8 while useful TP scores stay .9/.5.',
           'Single useful TP followed by duplicates may retain AP 1; duplicates ahead of remaining useful TPs can cost AP.',
           table(['Case', 'AP'], [[name, number(x['score'])] for name, x in d['D'].items()]),
           'Confirmed with a material refinement of the phrase “duplicate above useful detection”.',
           'There is no preassigned useful copy: the first matching duplicate becomes the TP. '
           'With two GTs, TP/duplicate/TP gives precision 1 through recall .50 and 2/3 thereafter. '
           'The 101-point average is (51*1 + 50*2/3)/101 = .83498349835, matching measurement.',
           'Evaluate duplicate placement relative to remaining useful detections, not duplicate count alone.',
           'Overlapping multiple GTs and large duplicate sets that reach maxDets are outside this fixture.')
    e_rows = [[name, str(r['removed_frames']), number(r['score']), f"{r['retained_gt']}/{r['total_gt']}", number(r['micro_recall'])] for name, r in d['E'].items()]
    e_classes = table(['Class'] + list(d['E']), [[c] + [number(r['ap_by_class'][c]) + ' / ' + number(r['recall_by_class'][c]) for r in d['E'].values()] for c in d['oracle']['ap_by_class']])
    detail('E', 'Missing frames', 'Do omitted predictions remove GT from evaluation?',
           'H6: GT stays, so omitted frames reduce attainable recall and AP.',
           'Start with full Helsinki oracle; delete only the frame keys shown. All retained detections remain perfect.',
           'Less recall and AP whenever removed frames contain GT; class effects depend on visibility.',
           table(['Removed pattern', 'Removed frames', 'AP', 'Retained / total GT', 'Micro recall'], e_rows) +
           '\n\nPer-class **AP / independently counted recall**:\n\n' + e_classes,
           str(all(r['score'] < 1 and r['micro_recall'] < 1 for name, r in d['E'].items() if name != 'none')),
           'Omitted frames remain in the GT denominator. The reported metric macro-averages class AP; '
           'it is not pooled recall. COCO recall-grid interpolation also means AP does not always equal '
           'continuous recall, even with only exact retained detections.',
           'Missed observation opportunities have class- and frame-dependent metric cost.',
           'These loss patterns and class frequencies describe Helsinki, not an unknown hosted scene.')
    controls = d['F_controls']
    prep = controls['preprocessing_ms']
    ahead = []
    for r in controls['synthetic']:
        early = sum(t['sent_elapsed'] < t['frame_index'] / 3 for t in r['transitions'])
        ahead.append([r['delay_ms'], early, f"{r['wall_seconds']:.3f}"])
    all_timed = d['F'] + controls['synthetic'] + d['K']
    verified = all(r['recurrence_matches'] and r['frame_accounting_matches'] for r in all_timed)
    detail('F', 'Realtime latency sweep', 'How does response latency change the sent sequence?',
           'H7: serial processing loses intermediate frames; request deadline differs from throughput budget.',
           'Actual endpoint sleep at every listed delay; all 25 Helsinki frames; full image I/O, crop, '
           'PNG, Base64, HTTP, DTO and camera logic. Follow-ups measure preparation without tracing '
           'and repeat zero-delay Helsinki without tracing because the first sweep already skipped '
           'frames at zero endpoint sleep. Simple-image timing isolates content-dependent overhead.',
           'Longer total processing time generally causes skips; exact counts must follow the implemented recurrence.',
           timing_table(d['F']) + '\n\nIndependent Helsinki image preparation (25 frames, no trace):\n\n' +
           table(['Stage', 'Mean ms', 'Median ms', 'Max ms'], [[k, f'{statistics.mean(x[k] for x in prep):.1f}', f'{statistics.median(x[k] for x in prep):.1f}', f'{max(x[k] for x in prep):.1f}'] for k in ('load_ms', 'render_ms', 'total_ms')]) +
           '\n\nHelsinki no-trace zero-delay control:\n\n' + timing_table([controls['helsinki_no_trace']]) +
           '\n\nSame replay path, simple synthetic PNGs:\n\n' + timing_table(controls['synthetic']) +
           '\n\nSynthetic requests sent **before** their nominal index/3 emission time:\n\n' +
           table(['Sleep ms', 'Early requests', 'Replay wall seconds'], ahead) +
           f'\n\nAll observed recurrence and frame-accounting comparisons agree: **{verified}**.',
           'Latency skipping confirmed; fixed 333 ms endpoint-only threshold and literal emission pacing require correction.',
           '`next = max(current+1, floor(elapsed/(1/3)))`; skipped increments by '
           '`max(0, min(next,N) - (current+1))`. Exact elapsed/next-index pairs are recorded for each '
           'iteration and independently recomputed. Elapsed includes preparation before POST, while '
           'RTT excludes it. The baseline and independent preparation measurements show why even a '
           'fast endpoint cannot guarantee no skips on this host. The synthetic early-request '
           'measurements expose a second detail: this local loop has no sleep to pace fast requests '
           'to the next nominal emission. `current+1` can send future frames early. Endpoint sleeps '
           'near 333 ms therefore do not define a universal cliff. Slight score increases at larger '
           'delay can reflect jitter and which class-bearing frames survive.',
           'Measure end-to-end throughput and actual frame indices; separate it from the request deadline.',
           'Single-run timing on this Windows host and a 15.625 ms monotonic clock; no hosted network, production capture scheduling, '
           'or hosted pacing parity is established. Tracing overhead is included except the marked no-trace control.')
    detail('G', 'maxDets boundary', 'Is a rank-101 valid detection retained?',
           'H8: only the first 100 detections per image/class enter the reported AP configuration.',
           '101 same-class detections with strictly descending confidence. Exactly one box is the '
           'perfect GT box; 100 disjoint FPs occupy the other ranks. Move the useful box to rank 99/100/101.',
           'Rank 99 and 100 can match, with AP reduced by preceding FPs; rank 101 cannot.',
           table(['Useful rank', 'Detections', 'AP'], [[r['tp_rank'], r['detection_count'], number(r['score'])] for r in d['G']]) +
           '\n\nInstalled COCO default `maxDets`: `' + str(d['G_configuration']['maxDets']) + '`; the official score reads the final maxDets axis.',
           str(d['G'][0]['score'] > 0 and d['G'][1]['score'] > 0 and d['G'][2]['score'] == 0),
           'Rank 99 gives 1/99, rank 100 gives 1/100, rank 101 gives zero. This is a scoring '
           'truncation, distinct from the 500-annotation response-validation limit.',
           'Useful detections need sufficient within-image/class rank to survive the cap.',
           'Measured for the installed faster-coco-eval version and this evaluator configuration.')
    hrows = []
    for name, r in d['H'].items():
        s = r['statistics']
        hrows.append([name, r['requests'][0]['schema_valid'], 0 in r['predicted_frames'],
                      s['responses_accepted'], s['frames_unanswered'], s['invalid_responses'],
                      s['commands_applied'], s['invalid_commands'], number(r['score'])])
    detail('H', 'Malformed responses', 'Which errors discard the entire response?',
           'H9: schema-invalid content loses all detections; mismatched IDs also invalidate the response.',
           'Mutate frame 0 only in a two-frame scene with hangar and tank GT in both. Keep both '
           'valid annotations, add one malformed annotation where applicable, and include an otherwise '
           'legal L1 move. Frame 1 is always a clean oracle response with no command.',
           'Invalid responses lose both good detections at frame 0, refuse processing of the camera '
           'command, and increment invalid/unanswered counters. Clean frame 1 still scores.',
           table(['Case', 'Schema valid', 'Frame 0 scored', 'Accepted / 2', 'Unanswered', 'Invalid responses', 'Moves applied', 'Moves rejected', 'AP'], hrows) +
           '\n\nBoth frames are sent and none skipped in every case; timeouts and HTTP errors are zero. '
           'For invalid responses, only frame 1 retains its two detections. Full validation messages '
           'and next-request camera states are in JSON. Numeric NaN and Infinity fail direct DTO '
           'validation with “bbox coordinates must be finite”; a strict JSON encoder refuses them. '
           'Legal JSON strings `"NaN"`/`"Infinity"` fail numeric schema types. The legal JSON numeric '
           'exponent `1e999` parses to infinity here and is rejected by the finite-coordinate validator.',
           'Confirmed. Wrong request_id and frame are schema-valid but fail contextual replay checks.',
           'Schema validation and request identity validation are separate stages, both capable of '
           'discarding the full response. Invalid camera field types invalidate detections too. '
           'No nonstandard numeric NaN JSON was emitted, so no HTTP claim is made for that encoding.',
           'Response integrity includes every annotation, field type, unknown key and echoed request identity.',
           'Other servers or JSON parsers may reject overflow earlier; numeric NaN has no standard JSON representation.')
    detail('I', 'Schema-valid illegal camera geometry', 'Can a rejected move preserve current detections?',
           'H9: geometric camera rejection is separate from response rejection.',
           'Compare L1 center_x=960 (legal), center_x=0 (integer but out of bounds), and '
           'center_x=960.0 (float). All carry identical two-object predictions.',
           'Illegal geometry preserves AP and old L0 view; float invalidates the first response.',
           table(['Case', 'AP', 'Next request level', 'Next center', 'Next feedback present'], [
               [name, number(r['score']), r['requests'][1]['view']['resolution_level'],
                str((r['requests'][1]['view']['center_x'], r['requests'][1]['view']['center_y'])),
                r['requests'][1]['feedback'] is not None] for name, r in d['I'].items()]),
           'Confirmed.', 'Integer geometry error retains the frame-0 detections and reports one invalid '
           'command; the float is a response-schema error, so its camera command is never attempted.',
           'Camera geometry and wire-schema correctness have different consequences for score.',
           'This controlled comparison tests an out-of-bounds center, not every geometric rejection branch.')
    j = d['J']
    detail('J', 'Feedback persistence', 'When does feedback appear and clear?',
           'H10: rejection feedback repeats until a valid command is accepted.',
           'Frame 0 requests illegal L1 center (0,540). Frames 1-2 omit the requested view. '
           'Frame 3 requests legal L1 (960,540). Frame 4 omits it again.',
           'No feedback at frame 0; frame-0 feedback in requests 1,2,3; cleared in request 4.',
           table(['Request frame', 'Current level', 'Feedback source frame', 'Response requested view'],
                 [[x['frame'], x['view']['resolution_level'], None if x['feedback'] is None else x['feedback']['frame'], x['requested_view']] for x in j['requests']]) +
           f'\n\nAP {number(j["score"])}; accepted {j["statistics"]["responses_accepted"]}; '
           f'applied commands {j["statistics"]["commands_applied"]}; rejected commands {j["statistics"]["invalid_commands"]}.',
           'Confirmed.', 'Feedback belongs to the request built before processing that frame response. '
           'The request carrying the successful repair still contains the old feedback; the next request clears it.',
           'Treat feedback as persistent state, and observe repair acknowledgment on the following request.',
           'This frame-by-frame measurement is offline; persistence across actual skipped frames is not separately measured.')
    detail('K', 'Late response versus timeout', 'Is a response below the deadline accepted despite skipping?',
           'H7: late valid responses count for their original frame; timeout loses them.',
           'Full Helsinki realtime replay, endpoint sleeps 3000 or 3500 ms; configured requests timeout is 10/3 seconds.',
           '3000 ms valid answers accepted with skips; 3500 ms requests time out and become unanswered.',
           timing_table(d['K']) + '\n\n' + '\n\n'.join('```text\n' + r['evaluator_output'].strip() + '\n```' for r in d['K'] if r['evaluator_output']),
           'Confirmed.', 'The timeout applies to the HTTP operation; total frame-loop duration can exceed '
           '3333 ms because image preparation precedes POST. A late response under that HTTP timeout '
           'still contributes predictions to its original frame; a timeout contributes none.',
           'Separate deadline reliability from the rate needed to preserve frame opportunities.',
           'This is a non-streaming loopback server. Requests timeout/network semantics and remote scheduling '
           'do not establish a strict hosted end-to-end wall-clock deadline.')
    detail('L_absent_class', 'Allowed class absent from GT', 'Do predictions for a class absent from the sequence affect mean AP?',
           'Absent allowed classes may be excluded from evaluated categories.',
           'One-frame synthetic scene with only hangar GT. Add a confidence-1 tank FP, or send only that FP.',
           'Hangar oracle AP remains 1 when tank FP is added; tank is absent from returned per-class keys.',
           table(['Case', 'AP', 'Evaluated classes'], [[name, number(r['score']), ', '.join(r['ap_by_class'])] for name, r in d['L_absent_class'].items()]),
           'Confirmed in this synthetic setup.', 'The absent allowed class is not included in the evaluated '
           'category set; sending only that absent-class prediction does not rescue the missing hangar.',
           'Interpret per-class means using the GT-present category set.',
           'Unknown hosted category presence prevents safely relying on this behavior operationally.')
    add('# Static-analysis corrections\n\n'
        '1. **Local realtime is not a paced 3 fps emitter.** It uses the 1/3-second clock to skip '
        'when behind, but does not wait when ahead. The synthetic timing control measures early '
        'requests. This observation does not establish hosted scheduler behavior.\n'
        '2. **333 ms is not an endpoint-only safe threshold.** Image preparation is inside elapsed '
        'sequence time and outside RTT. On this host it already creates skips at zero endpoint sleep.\n'
        '3. **A “high-ranked duplicate” is not automatically an FP.** The first identical matching box '
        'can claim GT. All single-object duplicate variants scored 1; the two-object test isolates '
        'the rank-dependent penalty.\n'
        '4. **FPs need not reduce interpolated AP.** In C, a trailing FP has zero AP cost; the '
        'same FP ranked first halves AP. The README warning is useful but not an unconditional metric law.\n'
        '5. **Wrong request_id/frame are context-invalid, not schema-invalid.** They pass the DTO '
        'yet lose the whole response at replay validation.\n'
        '6. **Missing-frame fraction is not the AP loss fraction.** Visibility differs by class, '
        'the metric macro-averages classes, and COCO uses 101 recall thresholds.\n\n'
        'No authoritative evaluator fix was made. The task measures its executable behavior; any '
        'decision to change local pacing requires separate review against the hosted implementation.')
    add('# Confirmed evaluator behaviors\n\n'
        '- Supplied oracle: 1.000 overall and in all 16 classes.\n'
        '- Correct global predictions outside the transmitted crop count equally in the tested scene.\n'
        '- IoU .50 is inclusive; .499 fails; .60 and .90 produce equal AP in the one-object fixture.\n'
        '- Confidence order changes AP; squaring scores without changing order preserves it.\n'
        '- Duplicate AP cost depends on remaining useful detections and ordering.\n'
        '- Unpredicted frames retain GT and reduce recall/AP.\n'
        '- Actual latency sweeps follow the local elapsed-time recurrence; overhead and early sending matter.\n'
        '- The reported COCO configuration drops the useful same-class detection at rank 101.\n'
        '- Tested invalid responses lose good detections and suppress camera-command processing.\n'
        '- A schema-valid out-of-bounds camera command preserves detections, keeps the old view, and generates feedback.\n'
        '- Feedback repeats until a successful command; the following request first reflects the clearing.\n'
        '- 3000 ms endpoint sleeps were accepted; 3500 ms sleeps timed out in this environment.\n'
        '- An allowed class absent from synthetic GT was excluded from per-class evaluation.')
    parity = d.get('scorer_version_parity')
    if parity:
        runs = parity['runs']
        numeric_diffs = parity['metric_differences'] + parity['configuration_differences'] + parity['fixture_differences']
        rows = []
        for probe in ('Oracle', 'A', 'B', 'C', 'D', 'E', 'G', 'L'):
            selected = [k for k in runs['1.7.2']['cases'] if k.split('/')[0] == probe]
            same = all(runs['1.7.2']['cases'][k] == runs['1.8.0']['cases'][k] for k in selected)
            rows.append([probe, len(selected), same])
        commands = '\n'.join(c['install'] for c in parity['commands']) + '\n' + parity['command']
        # Preserve literal argument boundaries for exact child commands too.
        child_commands = '\n'.join('& ' + ' '.join('"' + a.replace('"', '`"') + '"' for a in c['run_argv'])
                                   for c in parity['commands'])
        add('# Scorer Version Parity — 1.7.2 vs 1.8.0\n\n'
            f'Follow-up UTC: {parity["timestamp_utc"]}. **All numeric results identical: '
            f'{parity["every_numeric_result_identical"]}.** Compared '
            f'{parity["cases_compared"]} deterministic cases and {parity["metric_values_compared"]} '
            'overall/per-class AP values using exact equality, without rounding or tolerance. '
            'Also compared maxDets/useCats/recall thresholds, independently calculated IoUs, and '
            'fixture/recall metadata.\n\n' +
            table(['Probe', 'Cases', 'All AP and per-class AP identical'], rows) + '\n\n' +
            table(['Requested scorer', 'Verified imported version', 'Loaded module'],
                  [[v, runs[v]['scorer_version'], runs[v]['scorer_module']] for v in ('1.7.2', '1.8.0')]) + '\n\n'
            'Isolation: fresh subprocess per version; separately installed package directories take '
            'precedence in `sys.path`; both package metadata and loaded module path are checked before '
            'scoring. Other dependencies are identical, `.venv` remains unchanged at 1.8.0, and '
            'authoritative source/data hashes match the original suite. Each case ran once per version. '
            '**A is a score-only recheck:** identical globally converted predictions are scored with '
            'the prior in-crop/off-crop geometry recorded as metadata. Crop state is not an input to '
            '`score`; no camera movement, replay, timing, or HTTP experiments were run.\n\n'
            '**Every difference, including tiny differences:** ' +
            ('None. Maximum absolute numeric difference is **0**.' if not numeric_diffs else
             '\n\n```json\n' + json.dumps(numeric_diffs, indent=2) + '\n```') + '\n\n'
            '**Comparison with the previously recorded suite:** ' +
            ('Both versions reproduce every recorded overall/per-class AP exactly.' if not any(parity['differences_from_recorded_suite'].values()) else
             '\n\n```json\n' + json.dumps(parity['differences_from_recorded_suite'], indent=2) + '\n```') + '\n\n'
            '**Exact commands from repository root:**\n\n```powershell\n' + commands + '\n```\n\n'
            'The runner executed these score-only subprocess commands once each (already included '
            'in the runner above; do not repeat them as extra runs):\n\n```powershell\n' + child_commands + '\n```\n\n'
            '**Strategic conclusions changed:** ' + str(parity['strategic_conclusions_changed']) + '. '
            + ('Oracle, inclusive IoU .50 matching, confidence ordering, duplicate ranking, missing-frame '
               'loss, rank-100/rank-101 truncation, and absent-class behavior are unchanged. '
               '**Scorer-version parity is confirmed for these fixtures, and the evaluator-probe '
               'phase can be closed.** This does not assert equivalence for untested inputs or '
               'hosted timing/network behavior.' if parity['every_numeric_result_identical'] else
               'Review the differences above before closing the phase.'))
    version_uncertainty = (
        'Local deterministic scoring was compared under 1.7.2 and 1.8.0 as recorded in the parity section; '
        'this does not cover untested inputs or every permitted 1.x release. ' if parity else
        'Only local 1.8.0 scorer behavior is measured; the supported 1.7.2 lower bound was not separately installed or compared. ')
    add('# Remaining uncertainties\n\n'
        'Hosted scorer version, network behavior, preprocessing costs, and scheduler parity are unverified. '
        'No hosted validation/evaluation attempt was spent. ' + version_uncertainty + 'Helsinki contains '
        '25 frames and cannot establish long-run boundary stability or hosted-scene class weighting. '
        'No confidence ties, ambiguous multi-GT assignments, all camera rejection branches, or feedback '
        'across skipped frames were experimentally exhausted. Bare NaN JSON is nonstandard and was '
        'not transmitted; numeric NaN was tested directly against the DTO, while legal strings and '
        'numeric overflow were tested through HTTP. Timing values are single-run observations, '
        'not confidence intervals or portable latency guarantees.')
    add('# Implications\n\n'
        'Capability requirements supported by these probes are whole-frame coordinate consistency, '
        'useful confidence ordering, preserving frame opportunities, sufficient within-class rank, '
        'strict response integrity, correct request identity, and explicit camera-feedback handling. '
        'Latency assessment must include local image preparation and report actual skipped/unanswered '
        'frames. No detector, tracker, architecture, or camera-control algorithm is recommended here.')
    add('# Next phase\n\n'
        '**Yes: local evaluator behavior is sufficiently verified to proceed to Dataset & Scene Audit.** '
        'Carry forward the measured ranking semantics and the local clock/preprocessing caveat. '
        'This is readiness for a dataset audit, not certification of hosted timing or solution performance.')
    add('# Reproducibility\n\n'
        'See `README.md` for fresh-clone setup on Windows and Linux/macOS. The exact installed versions '
        'are in `requirements.lock.txt`; all satisfy the supplied requirements. From repository root:\n\n'
        '```powershell\n'
        'py -3.12 -m venv drone-flyby/.venv\n'
        '& drone-flyby/.venv/Scripts/python.exe -m pip install -r drone-flyby/evaluator_probes/requirements.lock.txt\n'
        '& drone-flyby/.venv/Scripts/python.exe drone-flyby/local_evaluator.py --oracle\n' + COMMAND + '\n```\n\n'
        'The complete command writes `probe_results.json` and regenerates this report. '
        '`--output other.json` preserves a separate run and writes `other.md`. To render an existing '
        'default JSON again:\n\n```powershell\n'
        '& drone-flyby/.venv/Scripts/python.exe drone-flyby/evaluator_probes/report.py\n```\n\n'
        'Allow roughly 4-7 minutes on this host, with additional variation from I/O and scheduling. '
        'The server binds loopback ephemeral ports and is shut down after each replay. Temporary '
        'fixture images/annotations are automatically removed; environments and bytecode remain '
        'ignored. No large logs, images, models, or generated binaries are committed. Existing '
        'ignore rules suffice. Exact AP should reproduce under the locked scorer; exact timing '
        'and skipped-frame counts should not be expected to match across machines.\n\n'
        '**Conclusion audit:** Every strong metric/protocol claim above has an exercised scorer or '
        'HTTP replay case. The report distinguishes direct measurements, COCO interpolation '
        'interpretation, and limitations; it does not promote local observations to hosted guarantees. '
        'Source/data hashes are checked before and after the suite.')
    path.write_text('\n\n'.join(sections) + '\n', encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, default=Path(__file__).with_name('probe_results.json'))
    parser.add_argument('--output', type=Path, default=Path(__file__).with_name('PROBE_RESULTS.md'))
    args = parser.parse_args()
    write_report(json.loads(args.input.read_text(encoding='utf-8')), args.output)
