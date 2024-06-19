import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import os
import country_converter as coco
from matplotlib.colors import ListedColormap
from matplotlib.axes import Axes
import seaborn as sns
from math import isclose
from src.preprocessing import main as preprocessing_main
from src.utils import (
    get_distance_matrix,
    plot_winkel_tripel_map,
    prepare_world,
)
import leidenalg as la
import igraph as ig
import infomap as imp

plt.style.use(
    "https://raw.githubusercontent.com/allfed/ALLFED-matplotlib-style-sheet/main/ALLFED.mplstyle"
)


class PyTradeShifts:
    """
    Class to build the trade matrix, calculate the trade shifts
    and plot them. This combines all the methods that are needed
    to easily run this from a jupyter notebook.

    Arguments:
        crop (str): The crop to build the trade matrix for.
        base_year (int): The base_year to extract data for. The trade communities
            are built relative to this year.
        percentile (float): The percentile to use for removing countries with
            low trade.
        region (str): The region to extract data for.
        testing (bool): Whether to run the methods or not. This is only used for
            testing purposes.
        scenario_name (str, optional): The name of the scenario to apply.
            If None, no scenario is applied.
        scenario_file_name (str, optional): The path to the scenario file.
            If None, no scenario is applied.
        only_keep_scenario_countries (bool, optional): Whether to only keep the
            countries that are in the scenario file or not. If True, only the
            countries in the scenario file are kept. If False, all the countries
            in the trade matrix are kept (but the scenario is only applied to the
            countries in the scenario file).
        with_preprocessing (bool, optional): Whether to run the preprocessing
            or not.
        countries_to_remove (list, optional): A list of countries to remove
            from the trade matrix. All other countries are kept.
            Mutually exclusive with countires_to_keep; takes priority.
        countries_to_keep (list, optional): A list of countries to keep in
            the trade matrix. All other countries are removed.
            Mutually exclusive with countries_to_remove; yields priority.
        keep_singletons (bool, optional): Whether to keep the communities
            with only one country or not. If False, these communities are
            removed.
        beta (float, optional): The parameter to use for the distance cost.
            When b == 0, there is no change to trade.
            When b > 0, the farther the two regions are the less trade between them.
            When b < 0, the farther the two regions are the more trade between them.
            It is recommended to keep this value low, e.g., in range [0, 2].
        make_plot (bool, optional): Whether to make the plot or not.
        shade_removed_countries (bool, optional): Whether to shade the countries
            that are removed from the trade matrix or not.
        cd_algorithm (str, optional): Community detection algorithm name.
            Supported names: `louvain`, `leiden`, `infomap`.
        cd_kwargs (dict, optional): Community detection algorithm keyworded argument.
            For possible kwargs see:
            `louvain`: https://networkx.org/documentation/stable/reference/algorithms/
                   generated/networkx.algorithms.community.louvain.louvain_communities.html
            `leiden`: https://leidenalg.readthedocs.io/en/stable/intro.html
            `infomap`: https://mapequation.github.io/infomap/python/infomap.html

    Returns:
        None
    """

    def __init__(
        self,
        crop: str,
        base_year: int,
        percentile=0.75,
        region="Global",
        testing=False,
        scenario_name=None,
        scenario_file_name=None,
        only_keep_scenario_countries=False,
        with_preprocessing=False,
        countries_to_remove=None,
        countries_to_keep=None,
        keep_singletons=False,
        beta=0.0,
        make_plot=True,
        shade_removed_countries=True,
        cd_algorithm="louvain",
        cd_kwargs={},
    ) -> None:
        # Save the arguments
        self.crop = crop
        self.base_year = "Y" + str(base_year)
        self.percentile = percentile
        self.region = region
        self.scenario_name = scenario_name
        self.scenario_file_name = scenario_file_name
        self.only_keep_scenario_countries = only_keep_scenario_countries
        self.countries_to_remove = countries_to_remove
        self.countries_to_keep = countries_to_keep
        self.keep_singletons = keep_singletons
        self.beta = beta
        self.make_plot = make_plot
        self.shade_removed_countries = shade_removed_countries
        self.cd_algorithm = cd_algorithm
        self.cd_kwargs = cd_kwargs
        # State variables to keep track of the progress
        self.prebalanced = False
        self.reexports_corrected = False
        self.no_trade_removed = False
        self.scenario_run = False
        # variables to keep track of the results
        self.trade_graph = None
        self.trade_matrix = None
        self.production_data = None
        self.threshold = None
        self.trade_communities = None

        # Don't run the methods if we are testing, so we can test them individually
        if not testing:
            self.run(with_preprocessing)

    def run(self, with_preprocessing: bool) -> None:
        """
        Executes the model.

        Arguments:
            with_preprocessing (bool): Whether to run the preprocessing
                or not.

        Returns:
            None
        """
        if with_preprocessing:
            preprocessing_main(
                "All_Data" if self.region == "Global" else self.region,
                self.crop,
                year=self.base_year,
            )
        # Read in the data
        self.load_data()
        # Remove countries with all zeroes in trade and production
        self.remove_net_zero_countries()
        # Prebalance the trade matrix
        self.prebalance()
        # Remove re-exports
        self.correct_reexports()
        # Set diagonal to zero
        np.fill_diagonal(self.trade_matrix.values, 0)
        # Remove countries
        if self.countries_to_remove is not None:
            self.remove_countries()
        elif self.countries_to_keep is not None:
            self.remove_countries_except()
        # Remove countries with low trade
        exiting = self.remove_below_percentile()
        if exiting is not None:
            print(exiting)
            return
        if self.scenario_name is not None:
            self.apply_scenario()
        # apply the distance cost only if beta != 0
        # for beta==0 there is no change in values
        # so there's no point in computing them
        if not isclose(self.beta, 0):
            self.apply_distance_cost()
        # Build the graph
        self.build_graph()
        # Find the trade communities
        self.find_trade_communities()
        if self.make_plot:
            # Plot the trade communities
            self.plot_trade_communities()

    def load_data(self) -> None:
        """
        Loads the data into a pandas dataframe and cleans it
        of countries with trade below a certain percentile.

        Arguments:
            None

        Returns:
            None
        """
        assert self.trade_matrix is None
        # Read in the data
        trade_matrix = pd.read_csv(
            "."
            + os.sep
            + "data"
            + os.sep
            + "preprocessed_data"
            + os.sep
            + f"{self.crop}_{self.base_year}_{self.region}_trade.csv",
            index_col=0,
        )

        production_data = pd.read_csv(
            "."
            + os.sep
            + "data"
            + os.sep
            + "preprocessed_data"
            + os.sep
            + f"{self.crop}_{self.base_year}_{self.region}_production.csv",
            index_col=0,
        ).squeeze()

        # ensure we do not have duplicates
        # duplicates are most likely a result of incorrect preprocessing
        # or lack thereof
        if not trade_matrix.index.equals(trade_matrix.index.unique()):
            print("Warning: trade matrix has duplicate indices, which will be removed.")
            print("The duplicates are the following: ")
            print(trade_matrix.index[trade_matrix.index.duplicated(keep=False)])

            trade_entries_to_keep = ~trade_matrix.index.duplicated(keep="first")
            trade_matrix = trade_matrix.loc[
                trade_entries_to_keep,
                trade_entries_to_keep,
            ]
        if not production_data.index.equals(production_data.index.unique()):
            print("Warning: trade matrix has duplicate indices, which will be removed.")
            print("The duplicates are the following: ")
            print(production_data.index[production_data.index.duplicated(keep=False)])
            production_data = production_data.loc[
                ~production_data.index.duplicated(keep="first")
            ]
        # remove a "not found" country
        # this would be a result of a region naming convention that
        # country_converter failed to handle
        if "not found" in trade_matrix.index:
            print("Warning: 'not found' present in trade matrix index")
            trade_matrix.drop(index="not found", inplace=True)
        if "not found" in trade_matrix.columns:
            print("Warning: 'not found' present in trade matrix columns")
            trade_matrix.drop(columns="not found", inplace=True)
        if "not found" in production_data.index:
            print("Warning: 'not found' present in production index")
            production_data.drop(index="not found", inplace=True)

        print(f"Loaded data for {self.crop} in {self.base_year}.")

        # Retain only the countries where we have production data and trade data
        countries = np.intersect1d(trade_matrix.index, production_data.index)
        trade_matrix = trade_matrix.loc[countries, countries]
        production_data = production_data.loc[countries]
        # Make sure this worked
        assert trade_matrix.shape[0] == production_data.shape[0]
        assert trade_matrix.shape[1] == production_data.shape[0]

        # Save the data
        self.trade_matrix = trade_matrix
        self.production_data = production_data

    def remove_net_zero_countries(self) -> None:
        """
        Return production and trading data with "all zero" countries removed.
        "All zero" countries are such states that has production = 0 and sum of rows
        in trade matrix = 0, and sum of columns = 0.

        Arguments:
            None

        Returns:
            None
        """
        assert self.no_trade_removed is False
        self.no_trade_removed = True

        countries_before = self.trade_matrix.index

        shape_before = self.trade_matrix.shape[0]

        # b_ signifies boolean here, these are filtering masks
        row_sums = self.trade_matrix.sum(axis=1)
        col_sums = self.trade_matrix.sum(axis=0)

        b_filter = ~(row_sums.eq(0) & col_sums.eq(0) & (self.production_data == 0))

        # Filter out the countries with all zeroes
        self.production_data = self.production_data[b_filter]
        self.trade_matrix = self.trade_matrix.loc[b_filter, b_filter]

        # print the number of countries removed
        print(
            f"Removed {shape_before - self.trade_matrix.shape[0]} "
            f"countries with no trade or production."
        )
        # print the names of the countries removed
        print(
            f"Removed countries: {countries_before[~b_filter].to_list()}"
            if not b_filter.all()
            else "No countries removed."
        )

    def prebalance(self, precision=10**-3) -> None:
        """
        This implementation also includes pre-balancing to ensure that countries don't
        export more than they produce and import.

        Arguments:
            precision (float, optional): Specifies precision of the prebalancing.

        Returns:
            None
        """
        assert self.prebalanced is False
        self.prebalanced = True

        # Make sure that the production data actually contains data
        assert self.production_data is not None
        assert self.production_data.sum() > 0

        # this is virtually 1:1 as in Croft et al.
        test = (
            self.production_data
            + self.trade_matrix.sum(axis=0)
            - self.trade_matrix.sum(axis=1)
        )
        while (test <= -precision).any():
            sf = (
                self.production_data + self.trade_matrix.sum(axis=0)
            ) / self.trade_matrix.sum(axis=1)
            multiplier = np.where(test < 0, sf, 1)
            self.trade_matrix = pd.DataFrame(
                np.diag(multiplier) @ self.trade_matrix.values,
                index=self.trade_matrix.index,
                columns=self.trade_matrix.columns,
            )
            test = (
                self.production_data
                + self.trade_matrix.sum(axis=0)
                - self.trade_matrix.sum(axis=1)
            )

        print("Prebalanced trade matrix.")

    def correct_reexports(self) -> None:
        """
        Removes re-exports from the trade matrix.
        This is a Python implementation of the R/Matlab code from:
        Croft, S. A., West, C. D., & Green, J. M. H. (2018).
        "Capturing the heterogeneity of sub-national production
        in global trade flows."

        Journal of Cleaner Production, 203, 1106–1118.

        https://doi.org/10.1016/j.jclepro.2018.08.267


        Input to this function should be prebalanced and have countries with all zeroes
        removed.

        Arguments:
            None

        Returns:
            None
        """
        assert self.prebalanced is True
        assert self.no_trade_removed is True
        assert self.reexports_corrected is False
        self.reexports_corrected = True

        # I know that the variable names here are confusing, but this is a conversion
        # of the original R code from Johanna Hedlung/Croft et al.. The variable names are the
        # same as in the R code and we leave them this way, so we can more easily
        # compare the two pieces of code if something goes wrong.
        self.trade_matrix = self.trade_matrix.T
        self.production_data = self.production_data.fillna(0)

        x = self.production_data + self.trade_matrix.sum(axis=1)
        try:
            y = np.linalg.inv(np.diag(x))
        except np.linalg.LinAlgError:
            print("Determinant=0 encountered in PyTradeShifts.correct_reexports().")
            print("Re-applying PyTradeShifts.remove_net_zero_countries().")
            self.no_trade_removed = False
            self.remove_net_zero_countries()
            print("Attempting to invert the matrix again.")
            x = self.production_data + self.trade_matrix.sum(axis=1)
            y = np.linalg.inv(np.diag(x))
        A = self.trade_matrix @ y
        R = np.linalg.inv(np.identity(len(A)) - A) @ np.diag(self.production_data)
        c = np.diag(y @ (x - self.trade_matrix.sum(axis=0)))
        R = (c @ R).T
        R[~np.isfinite(R)] = 0
        # we could consider parametrising this but Croft had this hard coded
        R[R < 0.001] = 0

        self.trade_matrix = pd.DataFrame(
            R, index=self.trade_matrix.index, columns=self.trade_matrix.columns
        )

        print("Corrected re-exports.")

    def remove_countries(self) -> None:
        """
        Removes countries from the trade matrix and production data.

        Arguments:
            None

        Returns:
            None
        """
        assert isinstance(self.countries_to_remove, list)
        # Convert the country names to the same format as in the trade matrix
        cc = coco.CountryConverter()
        self.countries_to_remove = cc.pandas_convert(
            pd.Series(self.countries_to_remove), to="name_short"
        ).to_list()

        # Take the index of the trade matrix and production data and remove all the countries
        # in self.countries_to_remove
        countries_to_keep = [
            country
            for country in self.trade_matrix.index
            if country not in self.countries_to_remove
        ]

        self.trade_matrix = self.trade_matrix.loc[countries_to_keep, countries_to_keep]

        self.production_data = self.production_data.loc[countries_to_keep]

        # print the number of countries removed
        print(
            f"Removed {len(self.countries_to_remove)} countries from the trade matrix."
        )

    def remove_countries_except(self) -> None:
        """
        Removes all countries from the trade matrix and production data except for the ones
        in self.countries_to_keep.

        Arguments:
            None

        Returns:
            None
        """
        assert isinstance(self.countries_to_keep, list)
        # Convert the country names to the same format as in the trade matrix
        cc = coco.CountryConverter()
        self.countries_to_keep = cc.pandas_convert(
            pd.Series(self.countries_to_keep), to="name_short"
        ).to_list()

        # Take the index of the trade matrix and production data and remove all the countries
        # in self.countries_to_remove
        keep = [
            country
            for country in self.trade_matrix.index
            if country in self.countries_to_keep
        ]

        self.trade_matrix = self.trade_matrix.loc[keep, keep]

        self.production_data = self.production_data.loc[keep]

        # print the number of countries retained
        print(f"Retained {len(self.countries_to_keep)} countries from the trade matrix")

    def remove_below_percentile(self) -> None | str:
        """
        Removes countries with trade below a certain percentile.

        Arguments:
            None

        Returns:
            None | str (upon failure)
        """
        # Make sure no threshold is calculated yet
        assert self.threshold is None

        # Calculate the percentile out of all values in the trade matrix. This
        # only considers the values above 0.
        try:
            threshold = np.percentile(
                self.trade_matrix.values[self.trade_matrix.values > 0],
                self.percentile * 100,
            )
        except IndexError:
            print(
                "There are no values in trade matrix within the specified percentile: %f"
                % self.percentile
            )
            return "Exiting PyTradeShift.remove_below_percentile()."
        # Set all values to 0 which are below the threshold
        self.trade_matrix[self.trade_matrix < threshold] = 0

        # b_ signifies boolean here, these are filtering masks
        row_sums = self.trade_matrix.sum(axis=1)
        col_sums = self.trade_matrix.sum(axis=0)

        b_filter = ~(row_sums.eq(0) & col_sums.eq(0))
        # Filter out the countries with all zeroes in trade
        self.trade_matrix = self.trade_matrix.loc[b_filter, b_filter]

        print(
            f"Removed countries with trade below the {int(self.percentile * 100)}th percentile."
        )

        # Save threshold for testing purposes
        self.threshold = threshold

    def apply_distance_cost(self) -> None:
        """
        Modifies the trade matrix to simulate transport costs.
        This stemms from the gravity law of trade where T ~ r^(-a).
        We modify the trade matrix by multiplying by r^(-b), effectively
        altering the parameter 'a' to be (a+b). 'a' is now going to be whatever
        follows from the data and 'b' is our control parameter (PyTradeShifts.beta).
        When b == 0, there is no change to trade.
        When b > 0, the farther the two regions are the less trade between them
        When b < 0, the farther the two regions are the more trade between them

        Arguments:
            None

        Returns:
            None
        """
        distance_matrix = get_distance_matrix(
            self.trade_matrix.index, self.trade_matrix.columns
        )
        if distance_matrix is None:
            print("Distance cost shall not be applied.")
            return
        # apply the modification of the gravity law of trade
        self.trade_matrix = self.trade_matrix.multiply(distance_matrix.pow(-self.beta))
        # diagonal will often be NaN here, so fill it with zeroes
        np.fill_diagonal(self.trade_matrix.values, 0)

        print(f"Applied distance cost with beta={self.beta}.")

    def apply_scenario(self) -> None:
        """
        Loads the scenario files unifies the names and applies the scenario to the trade matrix.
        by multiplying the trade matrix with the scenario scalar.

        This assumes that the scenario file consists of a csv file with the country names
        as the index and the changes in production as the only column.

        Arguments:
            None

        Returns:
            None
        """
        assert self.scenario_name is not None
        assert self.scenario_file_name is not None
        assert self.scenario_run is False
        self.scenario_run = True

        shape_before = self.trade_matrix.shape
        index_before = self.trade_matrix.index

        # Read in the scenario data
        scenario_data = pd.read_csv(
            "."
            + os.sep
            + "data"
            + os.sep
            + "scenario_files"
            + os.sep
            + self.scenario_file_name,
            index_col=0,
        ).squeeze()
        # Cast all the values to float
        scenario_data = pd.to_numeric(scenario_data, errors="raise")

        # make sure that this only contains numbers
        assert scenario_data.dtype == float

        assert isinstance(scenario_data, pd.Series)

        # Make sure that all the values are above -100, as this is a percentage change
        assert scenario_data.min() >= -100

        # Convert the percentage change to a scalar, so we can multiply the trade matrix with it
        scenario_data = 1 + scenario_data / 100

        # Make sure that all the values are above 0, as yield cannot become negative
        assert scenario_data.min() >= 0

        # Drop all NaNs
        scenario_data = scenario_data.dropna()

        cc = coco.CountryConverter()
        # Convert the country names to the same format as in the trade matrix
        scenario_data.index = cc.pandas_convert(
            pd.Series(scenario_data.index), to="name_short"
        )

        if self.only_keep_scenario_countries:
            # Only keep the countries that are in the trade matrix index, trade matrix columns and
            # the scenario data
            countries = np.intersect1d(
                np.intersect1d(self.trade_matrix.index, self.trade_matrix.columns),
                scenario_data.index,
            )
            self.trade_matrix = self.trade_matrix.loc[countries, countries]
            scenario_data = scenario_data.loc[countries]

            # Sort the indices
            self.trade_matrix = self.trade_matrix.sort_index(axis=0).sort_index(axis=1)
            scenario_data = scenario_data.sort_index()

            # Make sure the indices + columns are the same
            assert self.trade_matrix.index.equals(self.trade_matrix.columns)
            assert self.trade_matrix.index.equals(scenario_data.index)

            # Multiply all the columns with the scenario data
            self.trade_matrix = self.trade_matrix.mul(scenario_data.values, axis=0)
        else:
            # Multiply the trade matrix with the scenario data, but only for the countries
            # that are in the scenario data. Still keep all the countries in the trade matrix.
            # But first remove that are in the scenario data but not in the trade matrix, as
            # we are not interested in them.

            # Filter scenario data to include only countries present in the trade matrix
            scenario_data = scenario_data[
                scenario_data.index.isin(self.trade_matrix.index)
            ]
            # Add all the countries that are in the trade matrix but not in the scenario data
            # to the scenario data with a scalar of 1 (which means their production does not change)
            scenario_data = scenario_data.reindex(self.trade_matrix.index, fill_value=1)

            # Update trade matrix values based on scenario data (masking for missing values)
            self.trade_matrix = self.trade_matrix.mul(scenario_data, axis=0)

            # Assert index consistency
            assert self.trade_matrix.index.equals(self.trade_matrix.columns)

        # Should be the same as before
        assert shape_before == self.trade_matrix.shape
        assert index_before.equals(self.trade_matrix.index)

        print(f"Applied scenario {self.scenario_name}.")

    def build_graph(self) -> None:
        """
        Builds a directed and weighted graph from the trade matrix.

        Arguments:
            None

        Returns:
            None
        """
        assert self.trade_graph is None
        # only build the graph if all the prep is done
        assert self.prebalanced is True
        assert self.reexports_corrected is True
        assert self.no_trade_removed is True
        assert self.threshold is not None

        # Build the graph
        trade_graph = nx.from_pandas_adjacency(
            self.trade_matrix, create_using=nx.DiGraph
        )

        self.trade_graph = trade_graph

        print("Built trade graph.")

    def _get_communities(self) -> list[set[str]]:
        """
        Private function containing the appropriate set ups and execution
        for different community detection algorihthms.

        Arguments:
            None

        Returns:
            list[set[str]]: list containing sets of country names.
            Each set is a community.
        """
        match self.cd_algorithm:
            case "louvain":
                trade_communities = nx.community.louvain_communities(
                    self.trade_graph, **self.cd_kwargs
                )
            case "leiden":
                # temporary conversion to igraph as leidenalg is built on top of it
                trade_igraph = ig.Graph.from_networkx(self.trade_graph)
                # must specify partition type
                if "partition_type" not in self.cd_kwargs:
                    print("Partition type for Leiden method not specified.")
                    print("Using ModularityVertexPartition.")
                    self.cd_kwargs["partition_type"] = la.ModularityVertexPartition
                # get communities
                trade_communities = list(
                    la.find_partition(trade_igraph, **self.cd_kwargs)
                )
                # convert node IDs to country names
                trade_communities = [
                    {trade_igraph.vs[node_id]["_nx_name"] for node_id in community}
                    for community in trade_communities
                ]
            case "infomap":
                # create Infomap object
                im = imp.Infomap(**self.cd_kwargs)
                # import netowrkx trade graph
                im.add_networkx_graph(self.trade_graph)
                # run the algorithm
                im.run()
                # extract communities; using set() instead of unique()
                # to comply with other methods'  data types
                trade_communities = [
                    set(community_df["name"].values)
                    for _, community_df in im.get_dataframe(
                        columns=["name", "module_id"]
                    ).groupby("module_id")
                ]
            case _:
                print("Unrecognised community detection method.")
                print("Using Louvain algorithm with default parameters.")
                trade_communities = nx.community.louvain_communities(self.trade_graph)
        return trade_communities

    def find_trade_communities(self) -> None:
        """
        Finds the trade communities in the trade graph.

        Arguments:
            None

        Returns:
            None
        """
        assert self.trade_graph is not None
        assert self.trade_communities is None
        # Find the communities
        trade_communities = self._get_communities()
        # print number of communities found
        print(f"Found {len(trade_communities)} trade communities.")
        # Remove all the communities with only one country and print the names of the
        # communities that are removed
        if self.keep_singletons:
            print("Keeping communities with only one country.")
        else:
            for community in trade_communities:
                if len(community) == 1:
                    trade_communities.remove(community)
                    print(f"Removed community {community} with only one country.")

        self.trade_communities = trade_communities

    def _plot_trade_communities(self, ax: Axes) -> None:
        """
        Creates the plot of trading communities on the specified axis.

        Arguments:
            ax (Axes): the matplotlib axis on which to plot.

        Returns:
            None
        """
        assert self.trade_communities is not None
        # get the world map
        world = prepare_world()
        cc = coco.CountryConverter()
        # Create a dictionary with the countries and which community they belong to
        # The communities are numbered from 0 to n
        country_community = {}
        for i, community in enumerate(self.trade_communities):
            for country in community:
                # Convert to standard short names
                country_community[country] = i

        # Join the country_community dictionary to the world dataframe
        world["community"] = world["names_short"].map(country_community)

        # Plot the world map and color the countries according to their community
        cmap = ListedColormap(
            sns.color_palette("deep", len(self.trade_communities)).as_hex()
        )
        world.plot(
            ax=ax,
            column="community",
            cmap=cmap,
            missing_kwds={"color": "lightgrey"},
            legend=False,
        )

        plot_winkel_tripel_map(ax)
        # Add the countries which were removed from the trade matrix as shaded
        # countries
        if self.countries_to_remove is not None and self.shade_removed_countries:
            # Convert the country names to the same format as in the trade matrix
            self.countries_to_remove = cc.pandas_convert(
                pd.Series(self.countries_to_remove), to="name_short"
            ).to_list()
            world.loc[
                world["names_short"].isin(self.countries_to_remove), "geometry"
            ].plot(ax=ax, color="grey", hatch="xxx", alpha=0.5, edgecolor="black")

        # Add a title with self.scenario_name if applicable
        ax.set_title(
            f"Trade communities for {self.crop.lower()} with base year {self.base_year[1:]}"
            + (
                f"\nin scenario: {self.scenario_name}"
                if self.scenario_name is not None
                else ""
            )
            + (
                " with country subset"
                if self.countries_to_remove is not None
                or self.countries_to_keep is not None
                else ""
            )
        )

    def plot_trade_communities(self) -> None:
        """
        Plots the trade communities in the trade graph on a world map.

        Arguments:
            None

        Returns:
            None
        """
        _, ax = plt.subplots(figsize=(10, 6))
        self._plot_trade_communities(ax)

        # save the plot
        plt.savefig(
            "."
            + os.sep
            + "results"
            + os.sep
            + "figures"
            + os.sep
            + f"{self.crop}_{self.base_year}_{self.region}_"
            + (
                f"_{self.scenario_name}"
                if self.scenario_name is not None
                else "no_scenario"
            )
            + (
                "_with_country_subset"
                if self.countries_to_remove is not None
                or self.countries_to_keep is not None
                else ""
            )
            + "_trade_communities.png",
            dpi=300,
            bbox_inches="tight",
        )
