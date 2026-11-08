# Private-window candidate pair

See the [root README](../../README.md) for the canonical code map, selection
rationale, and complete phase classification.

| File | Private change from its SK4 parent | Status |
| --- | --- | --- |
| `private_closure_conditional_lean_w221.csv` | 40 disruption probabilities in weeks 222–234 are lowered to the model's independent-risk floor when the open MERROW closure does not occur. | **Recommended lean member of the final pair; private score unknown.** |
| `private_closure_conditional_insured_w221.csv` | 67 disruption probabilities in weeks 222–234 are raised to the model's closure-occurs branch. | **Recommended insured member of the final pair; private score unknown.** |

Both files have 260 valid rows and different private halves. Their public rows
match each other and their SK4 parents, so uploading both would spend two
public trials without a useful public-score comparison. The paired change is
intended for final private selection, where the better file counts.

The source is `../../experiments/private_closure_conditional.py`, which reads
the local SK4 snapshot in `../../methods/sk4_phase2/` and information through
week 221. It keeps prices, orders, and supplier shares unchanged from SK4.
The teammate's 48 private scenarios estimate **82.59** for the candidate
best-of-two versus **81.80** for its original pair, assuming a 40% closure
probability and the stated ban and closure outcomes. This is a scenario proxy,
not a leaderboard result. Across all ten historical live-closure cutoffs,
best-of-two score rose in four, fell by **0.03** in one, and was unchanged in
five under the local scorer; the largest gain, **+9.14**, was the actual
MERROW closure. The two real-closure origins overlap, and unwarned shocks
remain the main unknown. Evidence: `../../reports/private_closure_pair_backtest.csv` and
`../../reports/private_closure_backtest.csv`.

The observed **92.62** public file is `../submission92.62.csv`. Its private
rows still use the older pre-fix policy. Do not choose it for the final pair
based on its public score alone.
