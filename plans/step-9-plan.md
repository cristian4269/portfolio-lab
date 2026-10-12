# Step 9 plan: README, docs, final audit, publish

**Status: APPROVED by the owner on 2026-10-12, as written** (all eight decisions as
recommended). Read it with the blueprint's Step 9 section. Branch `step-9-publish` (first commit `b535815`: the blueprint marks Step 8 merged,
with its design pass, PR #12). Default model, 2 sessions, MVP.

## Grounding 2026-10-12

**The tree is almost clean.** A scan of every tracked file for local paths, the sibling
project, Tiingo, emails and secrets found three lines, all in `plans/portfolio-lab-blueprint.md`:
- line 127: the local Windows path of the repo (it shows the Windows user name);
- line 73 (invariant 3) and line 324 (Step 9's own task) name the sibling project's folder;
- Tiingo is named only as invariant 3's check (`git grep -il tiingo`) and in the acceptance
  criteria.

**The history holds nothing secret.**
- No file was ever committed and later deleted, except one bundle file under `app/data/`
  (derived results, allowed).
- No API key, token or password pattern appears in any commit.
- The Windows path and the sibling folder name sit in the scaffold commit `967e8db`.
- **12 merge commits carry the owner's personal email.** These are the PR merges done on
  GitHub (author `<personal gmail>`, committer GitHub). Every other commit uses the GitHub
  noreply address.
- The 12 PR descriptions mention no local path, sibling project or email. No issues exist.

**Hosting, re-checked today against Streamlit's docs (this corrects Step 8's D9 premise):**
- **An app from a private repo can be made public.** The docs say you "will need to make the
  app public if you want to freely share it" (Share → "Make this app public", or the app's
  Sharing settings). So G3 (deploy) no longer forces G4 (public repo). The two can be
  answered separately.
- A private repo still needs GitHub's `repo` OAuth scope for Streamlit, and only one private
  app is allowed at a time.
- **Dependencies:** Community Cloud uses the first file it finds, in the entrypoint's folder
  first and then the repo root, in this order: `uv.lock`, Pipfile, environment.yml,
  requirements.txt, pyproject.toml (Poetry). **The docs do not say** whether a `uv.lock`
  install includes the project itself (our src-layout package) or the `app` dependency group.
- **Resources** (dated February 2024): 0.078-2 CPU cores, 690 MB-2.7 GB RAM. An app with no
  traffic for 12 hours sleeps, and any visitor can wake it.

**The repo today:** private; description set; no homepage or topics. `pyproject.toml` has no
`[project.urls]`. The README is the Step 0 stub ("Work in progress"). `research/REPORT.md`
§12 still lists the dashboard as a next step.

## What gets done

**PR A (session 1): everything except the demo link**
1. **README.md**, recruiter first (D3).
2. **Screenshots** of the dashboard, light and dark, in `docs/img/` (D3).
3. **The scrub** of `plans/` (D2).
4. **Deploy dependencies:** `app/requirements.txt` exported from `uv.lock`, with a CI step that
   keeps it equal to the lock (D4).
5. **`pyproject.toml`:** `[project.urls]` (Repository, Report, Dashboard once known).
6. **`docs/career/`** (gitignored, stays local) (D7).
7. **Small doc fixes:** REPORT §12's dashboard line, NOTES.md entry, blueprint status and
   change log.
8. **Reviews and the fresh-clone audit** (D8), then push and the PR (asked).

**After PR A merges (session 2): the gates, then PR B**
9. **Ask G3 and G4** (D5), plus the repo metadata (D6).
10. **G4:** `gh repo edit --visibility public --accept-visibility-change-consequences`, then
    confirm CI is green on `main` and the badge renders.
11. **G3:** the owner signs in to Community Cloud and deploys with the settings I give (D5).
    I check the live app in the browser: every tab, the cold start and the numbers.
12. **PR B:** the demo link in the README and `pyproject.toml`, REPORT §12, the blueprint's
    acceptance criteria and status ("done"), the change log, NOTES. Then push (asked), merge
    (asked) and `/save-session`.

## Decisions for the owner

**D1. Keep the git history as it is (no rewrite).** *Recommended.*
- Rewriting (to drop the Windows path, or the email in the 12 merge commits) changes every
  commit hash after it. That includes the pushed tag `prereg-v1` (317d4c4), the holdout
  record `adccda8` and every hash cited in REPORT.md's timeline and the manifest.
- **The pre-registration's evidence is that timeline.** A rewritten history would break it.
- What stays visible: the Windows user name in one old line of the blueprint, and the
  personal email on 12 merge commits.
- **To stop future merges showing the email:** turn on GitHub's "Keep my email addresses
  private" (Settings → Emails). That is the owner's account setting; I won't change it.
- *Alternative:* rewrite with `git filter-repo`, re-tag and force-push. That breaks the
  pre-registration story. Not recommended.

**D2. Keep `plans/` public and scrub it in place.** *Recommended.*
- The blueprint and the nine step plans show the process (the decisions, the reviews, the
  amendments). A quant interviewer would read them.
- **The edits:**
  - the Windows path becomes "the repo root";
  - the sibling folder becomes "another local project";
  - invariant 3 keeps its Tiingo check (a vendor name; it shows the license care).
- *Alternative:* gitignore `plans/`. That hides the process, and the lines stay in history
  anyway.

**D3. README: the shape and the three headline results.** *Recommended as below.*
- **The top 30 seconds:**
  - the title and a one-line question;
  - badges (CI, Python 3.13, MIT);
  - the demo link (PR B);
  - a screenshot (`<picture>`, light and dark, like the report's figures);
  - three headline results.
- **The three results** (all from REPORT §1):
  1. **No pre-registered test rejects.** Neither minimum variance nor ERC beat 1/N after
     costs over the 188-month holdout. Every Holm-adjusted p is 1.00.
  2. **Minimum variance did its job.** Volatility fell from 16.7% to 13.8% and the worst
     drawdown from −36.7% to −32.7% (USD). The lower return cancelled the gain in Sharpe
     terms.
  3. **The tests are honest about power.** Over 188 months they detect only annual Sharpe
     differences of about 0.4-0.5 (minimum variance) and 0.12 (ERC), so a null was the
     expected outcome.
- **Then:**
  - the method in brief (pre-registration and its timeline, walk-forward t+1 execution,
    Ledoit-Wolf shrinkage, the LW 2008 test, VaR backtests, stress);
  - links to REPORT.md and PLAN.md;
  - the dashboard;
  - limitations;
  - how to run (from REPORT §13);
  - the project layout;
  - data sources and licenses (DATA_SOURCES.md);
  - "not investment advice".
- **Screenshot:** the Performance tab with the registered strip, at desktop width, light and
  dark, from the local app. The fallback, if the browser tool can't write the PNG, is the
  owner taking it.
- **Drafting:** routed to `documentation-writer` (Sonnet) with a self-contained brief, per
  ROUTING. I then check every number against REPORT.md.

**D4. Deploy dependencies: `app/requirements.txt` from the lock.** *Recommended.*
- **The file:** `uv export --no-dev --group app --no-hashes --format requirements-txt`, so the
  pins are exactly `uv.lock`'s and the project itself is installed.
- **Why `app/`:** a file in the entrypoint's folder wins over the root `uv.lock`, so Cloud
  does not depend on undocumented `uv.lock` handling.
- **A CI step** fails if the file differs from a fresh export, so it cannot drift from the
  lock.
- **Proved locally before the PR:**
  - a clean venv installs from the file, run from the repo root, with both pip and uv;
  - the app's AppTest smoke tests pass in that venv;
  - the app's memory stays well under Cloud's 690 MB floor.
- **Cloud settings:** Python 3.13 in the advanced settings; entrypoint
  `app/streamlit_app.py`.
- **The fallback**, if Cloud cannot install the project from a relative path: a small
  `sys.path` bootstrap at the top of `app/streamlit_app.py`. It goes in PR B, only if needed.

**D5. G3 and G4: the order, now that they are independent.** *Recommended:* scrub → merge PR
A → G4 (public) → G3 (deploy).
- With a public repo, Streamlit needs no `repo` scope over the owner's private repos.
- The demo link and the code become visible together.
- **I still ask both at once** (the blueprint's rule), but the owner can now say yes to one
  and no to the other. For example: deploy now from the private repo and make the app public,
  then decide G4 later.
- **The deploy itself is the owner's.** Signing in to share.streamlit.io and granting its
  GitHub access are account actions. I give the exact settings:
  - repo, branch `main`, entrypoint `app/streamlit_app.py`;
  - Python 3.13;
  - a subdomain such as `portfolio-lab` (if it is free).
  Then I verify the live app.

**D6. Repo metadata when it goes public** (asked with G3/G4).
- The homepage is the demo URL.
- The topics: `portfolio-optimization`, `backtesting`, `risk-management`,
  `value-at-risk`, `quantitative-finance`, `pre-registration`, `streamlit`, `python`.
- The existing description stays.

**D7. `docs/career/`** (local only, gitignored):
- `cv-bullet.md`: one bullet in English and one in French, each with a short and a long form.
- `pitch.md`: a 60-second pitch, plus a 15-second version.
- `interview-qa.md`: about 15 likely questions with answers. The blueprint's five (why
  Ledoit-Wolf, why a holdout, what Kupiec tests, t+1 execution, CAD hedging), plus:
  - why a null result is still a result (power, MDE);
  - Holm;
  - the LW 2008 test against Jobson-Korkie;
  - ERC;
  - FHS;
  - the FX timing;
  - what I would do next.

**D8. The fresh-clone audit**, in the scratchpad:
- `git clone` the branch, then `uv sync --all-groups && uv run pytest --cov` (≥ 80%; today
  99.96%).
- **A true cold start:** `reproduce --refresh` with `PORTFOLIO_LAB_CACHE` pointing at an
  empty folder. It downloads every source and runs the dev analysis.
  - If French has published a new vintage since 2026-10-10, the run **warns and names it**.
    That is the documented behaviour (REPORT §13), not a failure; I report it.
- **The holdout from a clean checkout:** `reproduce --holdout --rerun` against the existing
  frozen cache. Its tables must equal the recorded one-shot (as in 7b). The clone is thrown
  away, so the committed manifest is untouched.
- **Invariants 1-4:** greps and commands, run in the clone.
- **The README's links:** every relative link resolves. External links answer (one request
  each).

## Verification (before the PR)

- `uv run pytest --cov`, `ruff check`, `ruff format --check`, `mypy`, `uv lock --check`,
  `uv build`.
- The D4 venv proof and the D8 audit.
- **Invariants 1-6.** PLAN.md and `study.py` byte-identical to `prereg-v1`.
- **The scrub re-run:** no Windows path or sibling folder name in any tracked file.
- **Reviews:**
  - python-reviewer on the code-adjacent diff (CI step, `pyproject.toml`, requirements);
  - `/code-review` (high) on the branch.
  - Findings fixed, or reported if declined.

## Acceptance criteria this step closes (blueprint)

- [ ] CI green on `main`; the badge renders on the public repo.
- [ ] A live dashboard.
- [ ] README recruiter first, with the summary, the screenshot and the demo link.
- [ ] CV bullet and pitch in `docs/career/`.
- [ ] No raw data outside the allowed paths, no Tiingo data, no secrets, no local paths in
      the public tree.
- [ ] Fresh clone: `uv sync && uv run pytest` pass, coverage ≥ 80%; `reproduce` works and
      flags drift.

## Risks

| Risk | Mitigation |
|---|---|
| Cloud can't install the project from `app/requirements.txt` | Local pip and uv proof first; fallback `sys.path` bootstrap (D4) |
| A cold start on Cloud's CPU is well over 10 s | Measure it. If it's slow, `st.fragment` and lazier imports go to NOTES as follow-ups, not Step 9 scope |
| Memory near Cloud's 690 MB floor | Measure the local RSS before the PR |
| French publishes a new vintage | The audit expects and reports the drift warning; the holdout check uses the frozen cache |
| The personal email on 12 merge commits | D1: kept; GitHub email privacy for future merges (the owner's setting) |
| The subdomain is taken | Pick another at deploy time; the link goes in PR B |

## Gates in this step (each asked in this session)

1. This plan.
2. Pushing `step-9-publish` and opening PR A; merging it once CI is green.
3. **G3 + G4 together** (D5), with the D6 metadata.
4. Pushing PR B and merging it.
