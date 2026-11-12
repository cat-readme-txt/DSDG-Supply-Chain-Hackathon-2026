# Selected final private-window pair

See the [root README](../../README.md) for the canonical code map, selection
rationale, and complete phase classification.

| File | Private change from its SK4 parent | Status |
| --- | --- | --- |
| `private_closure_conditional_lean_w221.csv` | 40 disruption probabilities in weeks 222–234 are lowered to the model's independent-risk floor when the open MERROW closure does not occur. | **Selected final file 1 of 2; public 89.45, private score unknown.** |
| `private_closure_conditional_insured_w221.csv` | 67 disruption probabilities in weeks 222–234 are raised to the model's closure-occurs branch. | **Selected final file 2 of 2; public 89.45, private score unknown.** |

The user selected these **exact two CSVs** as the final pair and confirmed
that **each scored 89.45 publicly**. This records file selection and public
results, not a measured private score. Both files have 260 valid rows and
different private halves. Their public rows match each other and their SK4
parents, so the equal public scores cannot distinguish the private policies.

## Exact generation chain

| Final CSV | SHA-256 | Parent CSV and source strategy |
| --- | --- | --- |
| [`private_closure_conditional_lean_w221.csv`](private_closure_conditional_lean_w221.csv) | `69a39245b910959a7ec03897c799a7d03e856c9f1eecd075cef981693ce5eb77` | [`lean_pxlp_mkt_div_w221_pub.csv`](../../methods/sk4_phase2/submissions/lean_pxlp_mkt_div_w221_pub.csv), `lean_pxlp_mkt_div` |
| [`private_closure_conditional_insured_w221.csv`](private_closure_conditional_insured_w221.csv) | `764d501945396c187fa046b4f8dbe653ed2e069d4f1b539c8ad3e265fb6c7bae` | [`insured_hedged_mktcol_w221_pub.csv`](../../methods/sk4_phase2/submissions/insured_hedged_mktcol_w221_pub.csv), `insured_hedged_mktcol` |

Parent SHA-256 values are
`b9edeb78dd8ac3ff13ecacf74418fd14868b0fd85f4f84697a062f7f60ae9c40`
for lean and
`d495b94be27a92c32c6cbb2b55fce4ab2c00d3179a92f9e8d0a8ab5133aa0f71`
for insured. The copied SK4 package is based on Git commit `6c97ff9`.

Both parent strategies live in
[`methods/sk4_phase2/src/dsdg_supply_chain/strategies.py`](../../methods/sk4_phase2/src/dsdg_supply_chain/strategies.py).
They were run by `dsdg submit`, then the released-public rows were filled by
[`eda/33_fill_public.py`](../../methods/sk4_phase2/eda/33_fill_public.py).
The final transform is
[`experiments/private_closure_conditional.py`](../../experiments/private_closure_conditional.py)
with `--alpha 1` and the week-221 tables in
[`data/`](../../data) (the frozen source resolves its original
`data/Phase 2 data/` path through a symlink). It changes only private
disruption probabilities and preserves each parent's public rows and all
private prices, orders, and shares. The three model runs and exact formula are
documented in the [methods report](../../methods/README.md).

From the project root, run the following to rebuild the pair
in a temporary directory and compare every byte without overwriting the
selected CSVs:

```sh
python3 methods/check_provenance.py --final-only --python /path/to/python-with-dependencies
```

## Evidence and limits

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
