# Risk-aware portfolio lab

> Work in progress. Build plan: [plans/portfolio-lab-blueprint.md](plans/portfolio-lab-blueprint.md).

Does optimization beat naive 1/N diversification once you account for
estimation error, trading costs, and a Canadian investor's currency exposure?
This project builds portfolios (equal weight, minimum variance, risk parity,
factor tilts), backtests them walk-forward with a pre-registered holdout, and
validates their risk (VaR/ES with formal backtests, factor exposures, stress
tests).

## Quick start

```bash
uv sync
uv run pytest
uv run portfolio-lab --help
```

## Data

Only freely shareable sources, fetched at runtime and never committed. See
[DATA_SOURCES.md](DATA_SOURCES.md).

## License

Code: [MIT](LICENSE). Data keeps its own terms (see DATA_SOURCES.md).

Historical research for education; not investment advice.
