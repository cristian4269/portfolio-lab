# Step 10 plan: polish for quant research and risk recruiters

**Status: APPROVED 2026-10-12: all four PRs, in order. PR 1 in progress.** Phase 1 was a
read-only audit of every surface. The product-owner agent ranked the findings, and the main
session checked its corrections.

**The owner's decisions (2026-10-12):**
- The byline is "Cristian Ciorba, Finance student at HEC Montréal", with a LinkedIn link.
- LICENSE and pyproject keep the handle.
- The PDF goes in `docs/` and is attached to the release.
- No French one-pager.
- The AI sentence is used as drafted, without the unconfirmed "chose the question".

**Target readers.**
- **(a)** A recruiter for quant research / asset management or risk management, skimming for
  30 seconds.
- **(b)** The interviewer they forward it to, spending 10 minutes on rigor.

Never trade (b)'s trust for (a)'s attention.

**Constraints carried into every item:**
- PLAN.md and study.py stay identical to `prereg-v1`. No history rewrite (D1).
- No reported number changes. Every number traces to REPORT.md, and REPORT edits are dated.
- The null stays a null.
- New analysis is labelled exploratory / post-registration.
- Invariants 1-4 hold.
- The app stays well under Cloud's 690 MB.
- Every push, merge, tag, release and settings change needs the owner's yes.

## What the audit checked

- **The repo page:** About, topics, homepage, releases, social preview, the owner's profile
  and its other public repos.
- **Documents:** README.md, research/REPORT.md, pyproject.toml, LICENSE, ci.yml,
  .devcontainer, .streamlit/config.toml.
- **Code:** the package layout and public API. Docstrings: of 44 public defs without one,
  most are `@overload` stubs whose module docstring states the conventions.
- **Hygiene:** the scan is clean outside `plans/`. There, the only hits are invariant 3's own
  check and Step 9's scrub notes.
- **The live dashboard:**
  - Every tab renders, with no exception.
  - It reached charts about 9 s after navigation, warm.
  - At 375 px there is no horizontal scroll, the sidebar collapses, and the registered result
    is on the first screen.
- **CI on `main`:** green, including the `79c4fcc` run.

**Already good, no action:**
- The REPORT §1 executive summary.
- §10, "What failed or changed": a trust asset to point at.
- Thorough module docstrings with hand examples.
- Conventional commits with visible TDD RED/GREEN.
- Dark-mode figures and screenshots.

## Top 10 (in order)

1. **F02 + F07: who built it.**
   - README: name, HEC Montréal, one contact link, near the top.
   - REPORT: a byline and date.
2. **F04: a risk headline.**
   - README headline #4, in REPORT §8's table wording. Suggested: "Only filtered historical
     simulation gets the 95% exceptions at the right rate and without clustering (6 of 6
     series). At 99% no method passes conditional coverage in more than 1 of 6. Every series
     is in the Basel green zone."
3. **F03: "What this demonstrates".**
   - At most 8 bullets, each linked to evidence (a REPORT section, a test, a file).
   - It includes §10 as a trust asset.
4. **F25: say what "the assistant" is.**
   - One honest line in README Limitations and REPORT §2/§11: an AI coding assistant, what it
     did, and that the owner decided, approved and verified.
   - REPORT §11's "both authors" changes with it.
5. **N1: expected shortfall is shown but never backtested.**
   - Say so in README Limitations and REPORT §11, and name it in §12.
6. **F06 + N2 + F05: REPORT fixes (dated).**
   - §4 count: "sixteen besides 1/N (seventeen with it)".
   - §8 sentence to match its own table.
   - A risk and stress bullet in §1.
7. **F01 + F15: the dashboard links home.**
   - A "By <name> · Code · Report" row under the title.
   - Real links in the About tab, absolute URLs.
8. **F14: a model-comparison table on the Risk tab.**
   - Four methods × 95% and 99%: exceptions against expected, Kupiec, independence,
     conditional coverage, Basel zone, at the current settings, badged descriptive.
   - Keep the 99% default; showing both levels is the fix.
   - A test cross-checks the 1/N USD holdout row against REPORT §8.
9. **F11 + F08: the repo page and profile.**
   - About states the question, "pre-registered" and VaR.
   - Add four topics.
   - Profile name, bio and location; pin portfolio-lab.
10. **F23: a two-page research note (PDF) to attach to applications.**
    - From REPORT §1, §5 and §8 only.
    - Stamped with the commit and date. No new numbers.
    - HTML, printed to PDF with headless Edge (no pandoc here).

**Quick wins** (S effort, H or M impact): F02, F04, F03, F25, N1, F06, N2, F05, F01, F15, F11,
F08, F07, F12.

## All findings, ranked

| Rank | ID | Surface | Finding | Why a quant/risk reader cares | Proposed change | Effort | Impact | Risk to invariants | Owner-only? |
|---|---|---|---|---|---|---|---|---|---|
| 1 | F02 | README | No author line: README, REPORT, LICENSE and pyproject carry only the handle `cristian4269` | Nobody can contact or credit an anonymous artifact | Name, HEC Montréal, one link near the top | S | H | none | Supplies name and links |
| 2 | F04 | README | The risk-validation result is buried in Method; all three headlines are about Sharpe | Risk is the second pillar and the main hook for risk roles | Headline #4 in §8's table wording (above) | S | H | none (numbers from §8) | no |
| 3 | F03 | README | No section mapping the work to skills | (a) needs skills in 30 s; (b) needs each claim to point at evidence | "What this demonstrates", at most 8 linked bullets | S | H | none | no |
| 4 | F25 | README, REPORT | "The owner and the assistant" never says the assistant is an AI, what it did, or who verified | Interviewers will ask; disclosing first builds trust | One sentence in README Limitations and REPORT §2/§11; fix "both authors" | S | H | none (PLAN.md is untouched) | Decides the wording |
| 5 | N1 | README, REPORT | ES is listed (README) and shown (dashboard "Latest forecast"), but only VaR exceptions are backtested, and no limitation says so | A risk interviewer asks this first | One line in README Limitations and REPORT §11; name it in §12 | S | H | none | no |
| 6 | F06 | REPORT §4 | "Sixteen strategies in all" (17 with 1/N; README says sixteen besides 1/N; app says 17) | A counting slip hurts a rigor-themed repo | "Sixteen besides 1/N (seventeen with it)", dated | S | M | none | no |
| 7 | N2 | REPORT §8 | "No model's 99% exceptions are free of clustering" is stronger than the table, where FHS with 1000 days passes conditional coverage in 1 of 6 series | (b) will read the table against the sentence | Reword to match the table, dated; no number changes | S | M | none | no |
| 8 | F01 | Dashboard | No clickable link to the repo, REPORT or author; About says "Source code: the portfolio-lab repository." as plain text | A recruiter landing on the app from a CV can't reach the code | Absolute links in About and the header | S | M | low (live app) | no |
| 9 | F14 | Dashboard, Risk tab | One method at a time; default FHS at 99%, where §8 has FHS failing conditional coverage in 6 of 6 series, so the landing view (independence p 0.00) undercuts the headline | The key risk finding should be one glance, not several clicks | Comparison table, both levels, badged descriptive; keep defaults; test against §8 | M | H | low | no |
| 10 | F11 | Repo page | About is generic ("Build portfolios, backtest them honestly, ..."); topics already have pre-registration and value-at-risk | It is the first line of text on the repo page and in search | New About line; add replication, fama-french, expected-shortfall, asset-allocation | S | M | none | Applies via settings (owner's yes) |
| 11 | F23 | Docs | No short document to forward | A PDF is what reaches a hiring manager | Two-page note from §1, §5, §8; HTML to PDF | M | H | low (numbers must trace) | Name; repo or release asset? |
| 12 | F05 | REPORT §1 | "The answer" omits the risk and stress findings (§8) | (b) reads §1 first | A dated bullet reusing §8 numbers | S | M | none | no |
| 13 | F15 | Dashboard | The header has no byline | Credit and the way back to the code | Same change as F01 | S | M | low | Name |
| 14 | F08 | GitHub profile | No name, bio, location or link; no profile README; nothing pinned; public repos are portfolio-lab and a blackjack game | The profile is the page a recruiter lands on | Owner fills in the profile and pins portfolio-lab; Claude can draft a profile README (a new public repo) | S | M | none | Yes |
| 15 | F12 | CI, README | Coverage gate `fail_under = 80`, while actual is 99.96%; the README never states tests or coverage | Engineering signal, but only if enforced | Gate to 99, then "1,500+ tests, 99%+ branch coverage, enforced in CI" in the same PR | S | M | low | no |
| 16 | F07 | REPORT | No author or date byline | A research report carries its author | Byline and "last updated" | S | M | none | Name |
| 17 | F22 | Research | The prewhitened Parzen HAC was never checked against the authors' R code (REPORT §10 names the gap) | Closes a disclosed gap; strong for quant interviewers | Commit a script and reference values, not their code; any disagreement gets an erratum, never a quiet change | M | M | low | Installs R |
| 18 | F10 | Repo page | No release; version 0.1.0; only tag `prereg-v1` | A versioned release reads as finished work | Bump to 1.0.0 (pyproject, `__init__`, uv.lock, two asserts in tests/test_scaffold.py; the manifest keeps 0.1.0 as history); tag and release with notes and the PDF | S-M | M | low | Pushes the tag and release |
| 19 | F16 | Dashboard | Cold start after the 12-hour sleep unmeasured | A slow first load on a recruiter's click | Measure it and state it; `st.fragment` only if bad; no keep-alive bot | S / M | M | low | Times one cold start |
| 20 | F13 | Repo | No CITATION.cff | GitHub's "Cite this repository" button; research culture | Add it | S | L | none | Name |
| 21 | F09 | Repo page | Social preview is GitHub's automatic card | Matters when the link is posted on LinkedIn | Claude makes a 1280×640 PNG; the owner uploads it | S | L | none | Uploads |
| 22 | F21 | Research | No ES backtest (e.g. Acerbi-Székely) | Deep signal for market-risk roles; motivated by the elicitability literature (FRTB still backtests VaR) | Defer to an optional v1.1, labelled exploratory; N1 covers the honesty gap now | L | M | med (new code; study.py untouched) | no |
| 23 | F24 | Docs | No worked example | Shows the API in use | Defer. If done, it runs VaR/ES backtests, the Sharpe test and factor regressions on the bundle; the engine needs synthetic data (the industry matrix is not committed, by license) | M | L | none | no |
| 24 | F19 | CI | Runs on push and pull_request (duplicates on PR branches), no concurrency group; nothing installs app/requirements.txt in a clean venv (the AppTests already run on Linux CI) | Small engineering hygiene | Trigger fix, a concurrency group, an optional pip-from-requirements job | S | L | none | no |
| 25 | F27 | .devcontainer | `postAttachCommand` turns off Streamlit's CORS and XSRF (Streamlit's Codespaces template) | A security-minded reader may flag it | A comment: off only behind the Codespaces port proxy | S | L | none | no |
| 26 | F28 | GitHub account | 13 public commits carry the personal email (history kept, D1) | Privacy | "Keep my email addresses private" and block pushes that expose it | S | L | none | Yes |
| 27 | F29 | docs/career (local) | The French needs a native review; Q16 (AI use) to rewrite | Interview prep; keep it consistent with F25 | Owner's review | S | L | none | Yes |
| 28 | F20 | CI | No Dependabot | Little for a solo research repo | Actions only, if at all. A uv ecosystem would open PRs that fail the requirements-match gate and could drift reproduced numbers | S | L | low | no |
| 29 | F18 | Code | mypy excludes app/ (Altair keyword noise) | Cosmetic for both readers | Skip | M | L | low | no |
| 30 | F26 | Repo root | NOTES.md and plans/ sit at the root | The README already frames them as public on purpose | Skip: moving them breaks links | S | L | med | no |
| 31 | F17 | Code | `engine._cost_rate` imported privately by luck.py and dashboard.py | Invisible to both readers | Skip: engine.py is what the holdout ran; it would need a rerun and a fresh-clone audit | S | L | med | no |

## Skip or defer

- **F17:** touches the holdout's engine for no reader value.
- **F26:** breaks links.
- **F18:** cosmetic.
- **F20:** uv updates fight the requirements gate and could drift numbers.
- **F16:** no keep-alive bot (a grey area under Streamlit's fair use), and `st.fragment` only
  if the cold start is bad.
- **F21:** an optional v1.1.
- **F24:** defer, and reshape if done.
- **F29:** not a repo item.

## Suggested PR batches

1. **PR 1, docs: trust and message.** README and REPORT only, no code.
   - Items: F02, F07, F04, F03, F25, N1, F06, N2, F05.
   - Needs the owner's name, link and AI wording first.
   - After merge: F11 (the About line and topics, with the owner's yes) and F08 (the owner),
     so the repo page and README agree.
2. **PR 2, dashboard.** app/ only, plus AppTests.
   - Items: F01, F15, F14.
   - The merge redeploys the live app, so check it in the browser afterwards.
3. **PR 3, engineering polish.**
   - Items: F12 (the gate and the README stat together), F19, F27, F13, and the F10 version bump.
4. **PR 4, deliverables.**
   - Items: F23, and optionally a reshaped F24.
   - Then, with the owner's yes: tag `v1.0.0`, publish the release with the PDF attached,
     and F09.

Optional later, each on its own: F22 (R check), F21 (ES backtest, exploratory), F16 (measure).

## Owner-only actions

- Supply the name, affiliation and contact link to use. Decide whether LICENSE and pyproject
  carry the real name.
- Choose the AI-assistance wording (F25). Keep it consistent with docs/career Q16.
- In GitHub Settings:
  - apply the About line and topics (or say yes to `gh repo edit`);
  - upload the social preview;
  - set the profile name, bio and location;
  - pin portfolio-lab;
  - turn on "Keep my email addresses private".
- Approve a profile README (a new public repo), if wanted.
- Approve every push, merge, the `v1.0.0` tag and the release.
- Time one cold start, and visit the app within 12 hours before applications go out.
- Install R if F22 goes ahead.

## Decisions needed to start Phase 2

1. **Which items or batches** (default: PRs 1-3, then PR 4).
2. **Name and link(s)** for README, REPORT, the app header and CITATION.cff.
3. **The AI-assistance sentence** (F25).
4. **The PDF:** in `docs/` or only as a release asset.
5. **A French one-pager for Québec employers:** yes or no.
6. **Whether the blackjack repo stays public** beside this one (the owner's call; the
   suggestion is only to pin portfolio-lab).
