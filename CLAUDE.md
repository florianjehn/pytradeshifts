# PyTradeShifts — CLAUDE.md

## Project overview

PyTradeShifts is a network model that simulates how international agricultural
trade communities shift in response to yield changes (e.g. extreme climate
events, volcanic eruptions). It is based on
[Hedlung et al. (2022)](https://iopscience.iop.org/article/10.1088/1748-9326/aca68b)
and uses trade/production data from FAOSTAT.

Core workflow:
1. Load FAO trade matrix and production vector (tonnes, per country).
2. Correct re-exports so countries can only export what they produce
   (Croft et al. 2018 algorithm).
3. Optionally apply a yield-reduction scenario and/or a gravity-model distance
   penalty (`beta`).
4. Build a directed weighted graph (NetworkX) and detect trade communities
   (Louvain / Leiden / Infomap).
5. Run `Postprocessing` over a list of scenarios to compute metrics and
   generate world-map visualisations.

## Repository layout

```
src/
  model.py           # PyTradeShifts class — the main user-facing interface
  preprocessing.py   # Load & format FAO bulk-download ZIP/pickle files
  postprocessing.py  # Postprocessing class — metrics & plots
  utils.py           # Helpers: plot_node_metric_map, graph metrics, …
data/
  preprocessed_data/ # Ready-to-use trade & production CSVs (crop_year_region_*)
  data_raw/          # Raw FAO ZIP downloads
  scenario_files/    # Yield-reduction scenario CSVs (country → % reduction)
  stability_index/   # World Bank governance index used for node stability
tests/               # pytest test suite
scripts/             # Jupyter notebooks (usage examples)
```

## Running commands

The project uses [uv](https://docs.astral.sh/uv/) with a `.venv` at the repo
root. **Always activate or prefix commands with the venv; never use the system
Python or a globally installed pytest.**

```bash
# Run the full test suite
.venv/bin/pytest

# Run a specific test
.venv/bin/pytest tests/test_postprocessing.py::TestPostprocessing::test_compute_imports

# Start Jupyter
uv run jupyter notebook
```

## Key classes & functions

| Symbol | File | Purpose |
|---|---|---|
| `PyTradeShifts` | `src/model.py` | Load data, apply scenario, build graph, detect communities |
| `Postprocessing` | `src/postprocessing.py` | Compare scenarios, compute all metrics, generate plots |
| `plot_node_metric_map` | `src/utils.py` | Render a world map coloured by a per-country metric dict |
| `prep_fbs_consumption` | `src/preprocessing.py` | Load FAO Food Balance Sheet data → consumption Series |
| `format_prod_trad_data` | `src/preprocessing.py` | Load & unify production + trade data from pickles |
| `rename_countries` | `src/preprocessing.py` | Convert FAO M49 codes to short country names |

## Data conventions

- All trade/production quantities are in **tonnes**.
- FAO Food Balance Sheet (FBS) values are stored in **1 000 t** — `prep_fbs_consumption` converts them to tonnes automatically.
- Trade matrix: rows = exporters, columns = importers; column sum = total imports for that country.
- Country names use `country_converter` `name_short` format throughout.
- The first scenario passed to `Postprocessing` is always the **base scenario**.

## Adding a new plot / metric

The pattern used consistently across `postprocessing.py`:

1. Add a `_compute_*()` method that stores results on `self`.
2. Call it from `Postprocessing.run()` (or conditionally if optional data is required).
3. Add a `plot_*()` method that calls `plot_node_metric_map()` from `utils.py` for world-map views.
4. Add a test in `tests/test_postprocessing.py` checking shape and dtype of the result.

## FAO Food Balance Sheet (FBS) consumption data

To use `plot_import_change_as_fraction_of_consumption`:

1. Download the FBS bulk ZIP from https://www.fao.org/faostat/en/#data/FBS
2. Convert to pickle: `preprocessing.serialise_faostat_bulk("path/to/FoodBalanceSheets_E_All_Data.zip")`
3. Build a consumption Series:
   ```python
   from src.preprocessing import prep_fbs_consumption
   consumption = prep_fbs_consumption(
       "data/temp_files/FoodBalanceSheets_E_All_Data.pkl",
       item="Wheat and products",  # or "Maize and products", "Rice and products", "Soya beans"
       element="Food",             # human food use; alt: "Domestic supply quantity"
       year="Y2018",
   )
   ```
4. Pass it to `Postprocessing`:
   ```python
   pp = Postprocessing([base_scenario, other_scenario], consumption_data=consumption)
   pp.plot_import_change_as_fraction_of_consumption()
   ```
