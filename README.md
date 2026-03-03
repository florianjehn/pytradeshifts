# PyTradeShifts

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0) ![Testing](https://github.com/allfed/pytradeshifts/actions/workflows/testing.yml/badge.svg)

## What this is and what it can be used for

PyTradeShifts is using a network model to simulate how trade communities might shift due to changes in yield. This is based on the model presented in [Hedlung et al. (2022)](https://iopscience.iop.org/article/10.1088/1748-9326/aca68b). To make it possible to directly use yield reductions to change the trade flows, we have to account for re-exports. This is done using the re-export algorithm from [Croft et al. (2018)](https://www.sciencedirect.com/science/article/pii/S0959652618326180), as implemented in Hedlung et al. (2022), which allows countries only to export as much as they actually produce.

This allows us to change the trade flows, by changing the yields. Therefore, you can take any yield modelling results you have for different crops, feed them into this model and see how the trade flows change due to the changed yields. All this can be done via a single interface, which is explained below and in the [Jupyter notebooks of this repository](https://github.com/allfed/pytradeshifts/tree/main/scripts).

## Installation

This project uses [uv](https://docs.astral.sh/uv/) for environment and dependency management.

* Install uv if you don't have it yet:
  ```bash
  curl -LsSf https://astral.sh/uv/install.sh | sh
  ```

* From the root of the repository, create a virtual environment and install all dependencies:
  ```bash
  uv sync --extra dev
  ```
  This creates a `.venv` directory, installs the package in editable mode, and installs all dependencies including development tools (pytest, jupyter, ipykernel).

* To run the example Jupyter notebooks, start Jupyter via:
  ```bash
  uv run jupyter notebook
  ```
  Or, if you prefer to register a named kernel for use with an existing Jupyter installation:
  ```bash
  uv run python -m ipykernel install --user --name=pytradeshifts
  ```

## How to use this

The interface of the model is the class [PyTradeShifts in `model.py`](https://github.com/allfed/pytradeshifts/blob/main/src/model.py#L21). To work with the model you just have to create an instance of this class and specify all the keywords like you see fit (just click the link above to see all the keywords available):

`Wheat2018 = PyTradeShifts("Wheat", 2018)`

As you can see there are only two parameters that don't have default values. The crop you want to look at ("Wheat" in this case) and the base year you want to use (2018 in this case). This means every instance you create ia for a specific crop. The crops that are available is everything that is tracked in the [FAO database](http://www.fao.org/faostat/en/#data/TM) (though you likely want to stick with the bigger crops, to have more reliable results). The base year can be any year from 1961 (when the FAO started tracking) until today (if you have downloaded the most recent data). As trade partners tend to be sticky, changing this by a few years does not make a big difference in most cases. However, in general you want to use the most recent base year to better represent the present world.

### Running the model with a yield reduction scenario
One of the main tasks of this model is to try to understand how trade communities might change if yield is reduced, due to catastrophic events like extreme climate change or a large volcanic eruption. To do so you have to provide the model with a file the specificies how much the yield is reduced (in % reduction). These have to be put in the [scenario files folder](https://github.com/allfed/pytradeshifts/tree/main/data/scenario_files).  [An example](https://github.com/allfed/pytradeshifts/blob/main/data/scenario_files/ISIMIP_climate/ISIMIP_wheat_Hedlung.csv), of a scenario file is the reduction of wheat yields from the ISIMIP project. How to use a scenario is shown in [this Jupyter notebook](https://github.com/allfed/pytradeshifts/blob/main/scripts/ISIMIP_example.ipynb).

### Running the model with a long distance trade penality
In addition to the option to run the model with a yield reduction scenario, you can also apply an additional factor [`beta`](https://github.com/allfed/pytradeshifts/blob/main/src/model.py#L51) to specify how much long distance trade should be penalized. This is meant to symbolize the increased difficulties of trade after a catastrophe and follows a gravity model of trade approach.

## Getting the data

The repository already comes with pretty recent version of the FAO data, but if it is important to have the most recent one, you can download the data from the FAO: 1) [trade](http://www.fao.org/faostat/en/#data/TM) 2) [production](http://www.fao.org/faostat/en/#data/QC)

## Having problems?

If you encounter any issues, feel free to open an issue in the repository or contact florian@allfed.info

## Flowchart
```mermaid
%%{ init: { 'flowchart': { 'curve': 'natural' } } }%%
flowchart TD
    subgraph id_pre [Pre-processing]
    direction LR
    id_pre_3[Format trade matrix and production vector]
    end
    subgraph id_mod [Model]
    direction TB
        subgraph id_mod_sub_1 [" "]
        direction LR
        id_mod_4[Correct for re-exports]
        end
        id_mod_sub_1 --> id_mod_sub_2
        subgraph id_mod_sub_2 [" "]
        direction LR
        id_mod_6[Filter regions]
        end
        id_mod_sub_2 -->|optional| id_mod_sub_3
        id_mod_sub_2 --> id_mod_sub_4
        subgraph id_mod_sub_3 [" "]
        id_mod_8[Apply scenario]
        id_mod_9[Apply gravity model of trade]
        id_mod_8 --> id_mod_9
        end
        id_mod_sub_3 --> id_mod_sub_4
        subgraph id_mod_sub_4 [" "]
        id_mod_10[Build graph] --> id_mod_11[Find communities]
        end
    end
    subgraph id_pos [Post-processing]
    direction LR
    id_pos_1[Compute metrics] --> id_pos_2[Print]
    id_pos_1 --> id_pos_3[Plot]
    end
    FAO[(FAO data)] ==> id_pre
    id_pre ==> id_mod
    id_mod ==> id_pos
    id_pos ==o RES((Report))

    style id_pre fill:None,stroke:red,stroke-width:4px,stroke-dasharray: 5 5
    style id_mod fill:None,stroke:green,stroke-width:4px,stroke-dasharray: 5 5
    style id_pos fill:None,stroke:blue,stroke-width:4px,stroke-dasharray: 5 5
    style id_mod_sub_1 fill:None
    style id_mod_sub_2 fill:None
    style id_mod_sub_3 fill:None
    style id_mod_sub_4 fill:None
    style RES stroke:pink,stroke-width:4px
    style FAO stroke:yellow,stroke-width:4px
```
