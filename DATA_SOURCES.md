# Data sources

Every source used, its terms, and how to cite it. A source or series is used
only once its verdict here is "allowed". Terms pages were re-read on the
retrieval date shown; re-check them before publishing (blueprint step 9).

No data files are committed to this repository. Code downloads them at runtime
into a local cache (`~/.cache/portfolio-lab`, override with `PORTFOLIO_LAB_CACHE`).

## Sources

| Source | Terms page | Terms (key quote) | Retrieved | Citation | Verdict |
|---|---|---|---|---|---|
| Kenneth R. French Data Library | https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html | Copyright notice only: "Copyright Eugene F. Fama and Kenneth R. French". No explicit reuse terms. | 2026-10-02 | Kenneth R. French Data Library, https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html (CRSP vintage as stated in each file) | **Allowed under owner decision G1:** fetched at runtime, never committed, cited; only derived analytics published (no raw per-industry series shown or offered for download) |
| FRED, Federal Reserve Bank of St. Louis | https://fred.stlouisfed.org/legal/ | Public-domain series: "please cite the data source and acknowledge that you obtained the data from FRED." Pre-approval series "may only be used for non-commercial educational or personal use" without the holder's permission. No "disruptive" robots or scraping. | 2026-10-02 | "Source: Board of Governors of the Federal Reserve System (US) via FRED" | **Allowed per series** (see below); public-domain series only |
| Bank of Canada (Valet API) | https://www.bankofcanada.ca/terms/ | "You must attribute the Bank of Canada as the source of the content, and indicate if changes were made." Users may not "circumvent such limit or limits imposed by the Bank with respect to the number or frequency of requests made to a Bank Site, including [...] the Bank of Canada Valet API". | 2026-10-02 | "Source: Bank of Canada (Valet API). Converted from percent to decimal by portfolio-lab." | **Allowed**, with attribution and change note; cached, one request per series, polite delay |
| Statistics Canada (Web Data Service) | https://www.statcan.gc.ca/en/reference/licence | Statistics Canada Open Licence: users may "use, reproduce, publish, freely distribute, or sell the Information"; adapted use must say "Adapted from Statistics Canada, name of product, reference date. This does not constitute an endorsement by Statistics Canada of this product." | 2026-10-02 | "Adapted from Statistics Canada, Table 10-10-0122-01 Financial market statistics, last Wednesday unless otherwise stated, Bank of Canada (DOI 10.25318/1010012201-eng), <retrieval date>. This does not constitute an endorsement by Statistics Canada of this product." | **Allowed** |
| SEC EDGAR | https://www.sec.gov/edgar | - | - | - | Considered, not needed for this scope |

## Series

| Use | Series | Source | Frequency, coverage | Copyright line / licence | Verdict |
|---|---|---|---|---|---|
| Assets (main) | 12 Industry Portfolios, daily, value-weighted block (`12_Industry_Portfolios_daily_CSV.zip`) | French | Daily, 1926-07 on | G1 | Allowed (G1) |
| Assets (robustness) | 49 Industry Portfolios, daily, value-weighted block (`49_Industry_Portfolios_daily_CSV.zip`) | French | Daily, 1926-07 on | G1 | Allowed (G1) |
| Factors, US risk-free | Fama/French 5 Factors (2x3), daily (`F-F_Research_Data_5_Factors_2x3_daily_CSV.zip`): Mkt-RF, SMB, HML, RMW, CMA, RF | French | Daily, 1963-07-01 on | G1 | Allowed (G1) |
| Factors | Momentum factor, daily (`F-F_Momentum_Factor_daily_CSV.zip`) | French | Daily, 1926-11-03 on | G1 | Allowed (G1) |
| USD/CAD | `DEXCAUS`, CAD per 1 USD, H.10 noon buying rates in New York | FRED | Daily, 1971-01-04 on | "Public Domain: Citation Requested" | Allowed |
| CAD short rate | `v122531`, Treasury bills: 3 month, % per year | Statistics Canada table 10-10-0122-01 (data from the Bank of Canada) | Monthly, last Wednesday of the month; covers 1971 | Open Licence | Allowed |
| CAD 10-year yield (history) | `v122487`, Government of Canada marketable bonds, average yield, over 10 years, % per year | Statistics Canada table 10-10-0122-01 | Monthly, last Wednesday of the month; 1960 on | Open Licence | Allowed |
| CAD 10-year yield (daily) | `BD.CDN.10YR.DQ.YLD`, benchmark bond yield, 10-year, % per year | Bank of Canada Valet | Daily, 2001-01-02 on | Bank of Canada terms | Allowed |
| CAD 3-month T-bill (daily, cross-check) | `TB.CDN.90D.MID`, Treasury bills 3-month mid yield, % per year | Bank of Canada Valet | Daily, 2001-01-02 on | Bank of Canada terms | Allowed; used only to cross-check the monthly series |

**Not used.** `FXUSDCAD` (Valet, daily from 2017): `DEXCAUS` covers the whole sample, so no splice is needed. No pre-approval-only FRED series was needed; the check on each FRED series' copyright line is part of adding it here.

**Why Statistics Canada for Canadian rates.** Valet's daily T-bill and 10-year
series start in 2001, and its monthly copy of `V122531` starts in 2005. The CAD
view starts in 1971, so the long monthly history comes from table
10-10-0122-01 (the same Bank of Canada data, published by Statistics Canada).
Per the blueprint, the monthly short rate is forward-filled to daily; see
NOTES.md for the timing rule.
