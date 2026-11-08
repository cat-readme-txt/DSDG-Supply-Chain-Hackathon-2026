# EDA findings: Phase 1 (weeks 1–208)

Scripts: `eda/01_episodes.py` … `eda/04_market.py` (run with `uv run python eda/<file>`).
Outputs: `eda/episodes.csv`, `eda/country_runs.csv`, `eda/signal_news.csv`.

## 1. Ground truth is decodable

`disrupted[w, material] == 1` exactly when, for at least one of its suppliers:
`capacity < 0.85 × normal` OR `lead_time > base_lead` OR `tariff > base_tariff`.
This matches all 2,080 rows with 0 mismatches. So every individual shock (country, material, start, end, severity) is
recoverable from `supply_history` + `tariffs`.

## 2. Shock catalogue (weeks 1–208)

| Weeks | Who | Type | Effect | Benchmark uplift |
|---|---|---|---|---|
| 15–26 | VAL cobalt | tariff | tariff +0.30 | ~0 |
| 37–45 | KES steel | resource shortage | cap 0.33 | +22% |
| 70–71 | ATLANTIC lane (all HAL) | port strike | cap 0.8, lead +4 | ~0 |
| 78–80 | VAL rare earths | policy reversal (no warning) | cap 0.34 | +22% |
| 99–133 | KES all 7 materials | sanctions | cap 0.20, lead +3, tariff +0.21 on lithium/aluminum/semis from 122 | +8–20% |
| 101–103 | MERROW lane (SUN) | strait closure | cap 0.75, lead +3 | 0–13% |
| 109–113 | MERROW lane (SUN) | port strike | cap 0.8, lead +3 | 0–8% |
| 132–137 | ORI copper | export ban | cap 0.05 | +40% |
| 136–140 | VAL rare earths | export ban | cap 0.05 | +28% |
| 153–163 | VAL natural gas | export ban | cap 0.05 | +27% |
| 158–166 | HAL steel | resource shortage | cap 0.61 | +16% |
| 173–180 | MERROW lane (KES+SUN) | strait closure | cap 0.75, lead +3 | ~0–9% |
| 180–197 | VAL rare earths | tariff | tariff +0.12 | ~0 spot |
| 195–196 | HAL aluminum | policy reversal | cap 0.58 | +23% |

- About 14 shock events in 208 weeks, so roughly 1–2 per 13-week test window.
- Lane events (strike or closure) hit every country on the lane at once.
- Single-material capacity shocks raise *all* suppliers' spot prices, roughly in proportion to the capacity lost. Lane and
  tariff shocks barely move spot prices.
- Prices decay back over a few weeks after the shock ends.

## 3. News is the main leading signal

The news uses 90 headline templates. About 18 are routine noise and make up ~95% of articles. The signal templates form
scripted sequences:

| Type | Sequence → onset | Lead | Rumor resolution |
|---|---|---|---|
| Tariff | "trade dispute" → "confirms new duties" (body: "tariff of about X percent") / "front-load" → "take effect" | 1–2 wk (+`announced_rate` in tariffs.csv) | — |
| Shortage | "flag maintenance delays" ×2 → "rations water" / "accident halts" → "output forecast cut by about X percent" → "production falls" | 3–4 wk | — |
| Export ban | "debate limits" → "strategic priority" → "slower license approvals" / "drafts restrictions" → "signals curbs within weeks" / "traders brace" → "halts exports" | 1–4 wk | "officials rule out new restrictions" |
| Port strike | "vote on contract" → "congestion builds" → "union sets strike date" / "looks likely" / "shippers rush cargo" → "strike shuts ports" | 1–3 wk | "Deal reached, … disruption averted" |
| Strait closure | "tensions rise" → "war risk premiums" / "naval patrols" / "warning shots" → "traffic suspended after clashes" | 1–2 wk | "Calm returns near …" |
| Sanctions | "talks break down" → "prepare for sanctions" ×n → "vote scheduled" → "take effect" | ~7 wk | — |
| Policy reversal | "abruptly suspends … export licenses" in the same week | 0 | — |
| During / end | "continues", "no quick fix", "drags on" / "eases", "flows return to normal", "traffic returns" | — | — |

- Severity appears in the bodies ("forecast cut by about 67 percent" → cap 0.33; "tariff of about 12 percent" → +0.118).
- Duration is never stated. Observed durations: reversal 2–3, strike 2–5, closure 3–8, ban 5–11, shortage 9, tariff 12–18,
  sanctions 35.
- `tension_index` (macro) and `congestion_index` (shipping) rise with the news storyline, rumors included. They are
  weaker and noisier than the headlines, and they don't warn ahead of reversals.

### Open stories at the end of Phase 1 (week 208)

- **KES lithium:** "flag maintenance delays" in weeks 207 and 208, the opening of a shortage sequence. Expect a shortage
  around weeks 210–212. KES is the primary lithium supplier.
- **SOUTH lane (ORI):** "congestion builds" (207) and "labor talks drag on" (208), with SOUTH congestion at 38–39 against
  a normal ~30. A possible strike that would hit ORI lithium, cobalt, copper and natural gas. Unresolved.
- **NORTH lane (VAL):** tensions (203–205), then warning shots (206), then "Calm returns" (207). Likely a rumor that has
  ended.

## 4. Suppliers and the naive baseline

Every material has the same 3-tier structure (capacity as a multiple of mean demand, cost relative to the cheapest):

- Primary: 1.37× capacity, cost 1.00. KES for 7 materials, ORI for copper and natural gas, SUN for glass.
- Secondary: 0.79× capacity, cost 1.01–1.20.
- Tertiary: 0.49× capacity, cost 1.04–1.29. AUR (domestic rail, 1-week lead) is always the most expensive but is never
  hit by sea-lane shocks.

Naive submission: `order_units = demand`, 100% to the primary, `disruption_prob` = historical rate,
`price_forecast` = last benchmark (week 208).

## 5. Prices

- Price behaves like a random walk with ~2% weekly noise, loosely tied to a slow anchor. Carry-forward MAPE is ~4% at
  h=4 and ~5–9% at h=13. EWMA smoothing, 4-week futures and a linear trend don't beat carry-forward.
- `eda/10_price_drivers.py` tested macro (fx, PMI, CPI, policy rate, tension), freight and demand as drivers of
  future returns. All have |corr| < 0.06 and no out-of-sample skill.
- The one real signal is **mean reversion toward the 78-week average**: about half the gap closes within 13 weeks
  (correlation −0.39 at h=13). Applied at half strength it gives about +0.05 MAPE skill near-term and +0.09 further out.
- Shock uplifts depend on the shock type rather than the size of the capacity loss: ban ~+23%, shortage, reversal and
  sanctions ~+12–14%, lane closures and strikes ~0. The jump lands in the onset week with no build-up, and it fades
  within about a week of the shock ending.
- Landed cost ≈ spot × (1 + tariff) + a freight term that depends on the lane and material.

## Implications for the pipeline

1. **Disruption probability:** a rule-based news state machine per (country/lane, material), giving P(onset in the next k
   weeks) and P(still active). Blend it with a per-type duration prior and a base hazard for unwarned shocks.
2. **Price:** carry-forward × expected shock uplift (by type and severity) × decay.
3. **Policy:** this is 70% of the score. We need a local copy of the scoring simulator, then optimize order size
   (buffering ahead of predicted shocks) and shares (rerouting away from the hit supplier, plus the diversification
   bonus).
4. **Phase 2 (Sat 12 PM):** weeks 209–221 become labels. Re-check the open stories and re-tune for weeks 222–234.

## 6. Fresh-eyes re-analysis (Round 6, Sat ~4 AM): market signals and cost structure

Scripts: `eda/28_market_signals.py` (tables below), `eda/29_state_report.py [origin]` (what to plan for at an origin).

**Futures anticipate real price-moving shocks, not rumors.** The basis (futures_4w / spot − 1) on the supplier
about to be hit ramps up from the week the news story starts and collapses to ~0 at onset:

| Shock | Story weeks before onset | Basis path (%) |
|---|---|---|
| KES steel shortage (37) | 33–36 | 2.3, 5.4, 9.0, 11.7 |
| KES cobalt shortage (117, under sanctions) | 113–116 | 2.3, 7.5, 8.7, 13.5 |
| ORI copper ban (132) | 128–131 | 8.0, 13.6, 18.1, 24.4 |
| VAL rare earths ban (136) | 133–135 | 6.6, 14.5, 25.0 |
| VAL natural gas ban (153) | 151–152 | 7.4, 22.4 |
| HAL steel shortage (158) | 154–157 | 3.6, 5.2, 12.0, 12.7 |
| VAL natural gas ban **rumor** (48–53, called off) | 48–53 | −1.3, 0.0, −0.6, −1.1, −0.7, −0.2 |
| KES lithium shortage story (open at 208) | 207–208 | 2.6, 4.1 |

- Calm basis: sd 1.2%, 99.5th percentile 3.2%, max 4.9%; above 4% in 0.13% of calm rows. So a basis above ~3.2% is
  a >2.5σ event; every real story crossed it within 2 weeks, the rumor never did. One news body even says
  "Futures for <material> rose on the news."
- No ramp before reversals (unwarned), lane shocks, tariffs or the sanctions (no spot uplift to anticipate). Futures
  never anticipate a shock's **end**. In calm weeks the basis has no predictive power (corr with the next 4-week
  return ≈ 0), which is why futures looked useless as a price forecaster (P5).

**Lane stories: congestion and tension separate real from rumor only in the last week.**
Real: ATLANTIC strike congestion 40 → 57 → onset; MERROW strike 41, 46, 52 → onset; MERROW closure 38, 55 → onset
(lane-country tension 36, 54). Rumors peak at 40–46 (tension ≤ 38). The open SOUTH strike story at 208 sits at 39, 39.

**Other structure (verified, exact or near-exact):**
- Landed cost = spot × (1 + tariff) + freight. Freight is a constant per (material, country) on NORTH/SOUTH/RAIL and
  ∝ container rate on MERROW/ATLANTIC (~6% of landed cost on MERROW, ~4% on ATLANTIC). The MERROW container rate goes
  ~1,800 → ~5,300 in closures (so KES/SUN landed cost +10–12%) and ~2,800 in strikes.
- Spot uplifts hit all three suppliers of the material in the onset week and vanish the week after the shock ends.
  News never signals the end ("ongoing" articles to the last week; "recovered" exactly one week after).
- The "35-week KES sanctions" (99–133) is two overlapping sanctions stories (second one effective 118); the +3-week
  lead inside it comes from the MERROW closure (101–103) and strike (109–113). The duration model already shrinks
  sanctions to a ~18-week median.
- Capacity noise outside shocks is i.i.d. (0.91–1.01 × normal, no autocorrelation); lead times have no noise; tariff
  announcements appear 2–8 weeks ahead with the exact rate and week.
- Prices: mean reversion is real at long horizons (variance ratio 0.88 at 4 weeks, 0.60 at 13, 0.45 at 26). A
  full-strength pull to the 78-week anchor beats the half-strength one in a walk-forward on shock-free weeks
  (MAPE skill vs carry-forward +0.10 vs +0.06 at h = 13). Supplier spots share a common factor (corr 0.81–0.89); no
  lead-lag across suppliers or materials; materials are independent.

**Not useful (tested):** news source (it only reflects geography: local papers cover their own country), the
"<material> prices little changed" headlines (appear during shocks too), demand (independent of shocks), macro
except tension/FX before sanctions (FX −11% and tension 17 → 51 over the 10 weeks before the first sanctions).

**Gap found in the pipeline:** the ordering cost model (`policy.supplier_state`) holds the origin week's landed cost
flat for the whole window (only announced tariffs change it), in every scenario. It doesn't see predicted shock
uplifts, the end of an ongoing uplift, or freight spikes, so it can't pre-buy before a warned price jump the way the
hindsight plan (F's denominator) does.
