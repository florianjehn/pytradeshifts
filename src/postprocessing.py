import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from src.model import PyTradeShifts
from src.utils import (
    all_equal,
    get_degree_centrality,
    get_entropic_degree,
    jaccard_index,
    plot_node_metric_map,
    get_right_stochastic_matrix,
    get_stationary_probability_vector,
    get_entropy_rate,
    get_dict_min_max,
    get_graph_efficiency,
    get_stability_index,
    get_distance_matrix,
    get_percolation_threshold,
    fill_sector_by_colour,
)
import numpy as np
import pandas as pd
from operator import itemgetter
from functools import reduce
import seaborn as sb
from scipy import stats
import os
from pathlib import Path
from datetime import datetime, timezone


plt.style.use(
    "https://raw.githubusercontent.com/allfed/ALLFED-matplotlib-style-sheet/main/ALLFED.mplstyle"
)


class Postprocessing:
    """
    This class is used to postprocess the results of the scenarios.
    It works for an arbitrary number of scenarios.
    It compares the scenarios and generates a report of the results.

    Arguments:
        scenarios (list): List of scenarios to compare. Each scenario should be an instance
            of PyTradeShifts. The first scenario in the list is considered the
            'base scenario' and will be used as such for all comparison computations.
        anchor_countries (list, optional): List of country names to be used as anchores in plotting.
            The expected behaviour here is that the specified countries will retain
            their colour across all community plots.
        normalisation (str | None, optional): The normalisation version in the network efficiency
            metric. Possible values are `weak`, `strong` and None.
            If None, the metric is not normalised, `strong` uses an ideal flow matrix
            as the normalisation factor, and `weak` the mean of actual and ideal flow.
        stability_index_file (str, optional): Path to file containing stability metric
            for each country. The format is a two columns .csv file with columns: [country, index].
            By default it leads to a World Bank data based index we provide in the repository.
        gamma (float, optional): distance scaling factor in node stability computation.
            Node stability depends on the distances to other nodes.
            This distance factor is d(n,m)^(-gamma), where d(n, m) is the distance between
            nodes `n`, `m`. Default gamma = 1.0.
        random_attack_sample_size: (int, optional): Specifies the number of times
            to conduct random attack on the network. The higher the number the lower
            the uncertainty but higher computation time. Must be >= 2.
        testing (bool, optional): Whether to run the methods or not. This is only used for
            testing purposes.

    Returns:
        None
    """

    def __init__(
        self,
        scenarios: list[PyTradeShifts],
        anchor_countries: list[str] = [],
        normalisation="weak",
        stability_index_file=f"data{os.sep}stability_index{os.sep}worldbank_governence_indicator_2022_normalised.csv",
        gamma=1.0,
        random_attack_sample_size=100,
        testing=False,
    ):
        self.scenarios = scenarios
        # we could make this user-specified but it's going to make the interface
        # and the code more complicated, let's just inform in the docs
        # that the first passed scenario is considered the base
        self.anchor_countries = anchor_countries
        # check if community detection is uniform for all objects
        # there might be a case where it is desired so we allow it
        # but most times this is going to be undesirable hence the warning
        if not all_equal((sc.cd_algorithm for sc in scenarios)):
            print("Warning: Inconsistent community detection algorithms detected.")
        if not all_equal((sc.cd_kwargs for sc in scenarios)):
            print("Warning: Inconsistent community detection parameters detected.")

        self.normalisation = normalisation
        self.stability_index_file = stability_index_file
        self.gamma = gamma
        self.random_attack_sample_size = random_attack_sample_size
        assert self.random_attack_sample_size >= 2
        if not testing:
            self.run()

    def run(self) -> None:
        """
        Executes all computation required for reporting.

        Arguments:
            None

        Returns:
            None
        """
        print("Starting postprocessing computations...")
        # in order to compute matrix distances we need matrices to be
        # of the same shape, this is a helper member variable that allows that
        self.elligible_countries = [
            set(scenario.trade_graph.nodes()).intersection(
                self.scenarios[0].trade_graph.nodes()
            )
            for scenario in self.scenarios[1:]
        ]
        if self.anchor_countries:
            self._arrange_communities()
        self._compute_imports()
        self._compute_imports_difference()
        self._add_weight_reciprocals()
        self._find_community_difference()
        self._compute_frobenius_distance()
        self._compute_entropy_rate_distance()
        self._compute_stationary_markov_distance()
        self._format_distance_dataframe()
        self._compute_centrality()
        self._compute_global_centrality_metrics()
        self._compute_community_centrality_metrics()
        self._compute_community_satisfaction()
        self._compute_community_satisfaction_difference()
        self._compute_efficiency()
        self._compute_clustering_coefficient()
        self._compute_betweenness_centrality()
        self._compute_within_community_degree()
        self._compute_participation()
        self._compute_node_stability()
        self._compute_node_stability_difference()
        self._compute_network_stability()
        self._compute_entropic_out_degree()
        self._compute_percolation_threshold()

    def _compute_imports(self) -> None:
        """
        Calculates how much imports each country receives from each other country.
        This is done for each scenario.

        Arguments:
            None

        Returns:
            None
        """
        self.imports = []
        for scenario in self.scenarios:
            imports = {}
            # Go through all columns in the trade matrix. Each column represents the imports of a country.
            for country in scenario.trade_matrix.columns:
                # Sum the imports from all other countries
                imports[country] = scenario.trade_matrix[country].sum()
            self.imports.append(imports)

    def _compute_imports_difference(self) -> None:
        """
        Computes the decrease in imports for each country in each scenario in percentage.
        The decrease is calculated relative to the base scenario.

        Arguments:
            None

        Returns:
            None
        """
        self.imports_difference = []
        for imports in self.imports[1:]:
            imports_difference = {}
            for country in imports:
                if country not in self.imports[0]:
                    print(f"Warning: {country} not found in the base scenario.")
                    imports_difference[country] = np.nan
                else:
                    if self.imports[0][country] == 0:
                        imports_difference[country] = 0
                    else:
                        imports_difference[country] = (
                            (imports[country] - self.imports[0][country])
                            / self.imports[0][country]
                        ) * 100
            self.imports_difference.append(imports_difference)

    def plot_imports_difference(
        self,
        figsize: tuple[float, float] | None = None,
        shrink=1.0,
        file_path: str | None = None,
        file_format="png",
        dpi=300,
        **kwargs,
    ) -> None:
        """
        Plots the imports difference for each scenario on a world map with the countries
        coloured by the decrease in imports in percentage.

        Arguments:
            figsize (tuple[float, float] | None, optional): The composite figure
                size as expected by the matplotlib subplots routine.
            shrink (float, optional): Colour bar shrink parameter.
            file_path (str | None, optional): Path to where the image file
                should be saved to. If `None` no file shall be produced.
            file_format (str, optional): File extension to use when
                saving plot to file.
            dpi (int, optional): DPI of the saved image file.
            **kwargs (optional): Any additional keyworded arguments recognised
                by geopandas plot function.

        Returns:
            None
        """
        assert len(self.scenarios) > 1
        _, axs = plt.subplots(
            len(self.scenarios) - 1,
            1,
            sharex=True,
            tight_layout=True,
            figsize=(
                (5, (len(self.scenarios) - 1) * 2.5) if figsize is None else figsize
            ),
        )
        # if there are only two scenarios axs will be just an ax object
        # convert to a list to comply with other cases
        try:
            len(axs)
        except TypeError:
            axs = [axs]
        for ax, (idx, scenario) in zip(axs, enumerate(self.scenarios[1:])):
            plot_node_metric_map(
                ax,
                scenario,
                self.imports_difference[idx],
                "Import relative difference [%]",
                shrink=shrink,
                **kwargs,
            )
        if file_path:
            plt.savefig(
                f"{file_path}{os.sep}import_diff.{file_format}",
                dpi=dpi,
                bbox_inches="tight",
            )
        else:
            plt.show()

    def _add_weight_reciprocals(self) -> None:
        """
        Adds a new edge attribute `1/weight` that is the reciprocal of the weight
        attribute, for each scenarios' graph.
        This is later used to compute betweenness and efficiency.

        Arguments:
            None

        Returns:
            None
        """
        for scenario in self.scenarios:
            nx.set_edge_attributes(
                scenario.trade_graph,
                values={
                    (u, v): d["weight"] ** -1 if d["weight"] != 0 else np.inf
                    for (u, v, d) in scenario.trade_graph.edges(data=True)
                },
                name="1/weight",
            )

    def _compute_frobenius_distance(self) -> None:
        """
        Computes the Frobenius distance between the base scenario graph
        and each other scenario as the Frobenius norm of A-A',
        where A is the adjacency matrix of the base scenario, and A' of some
        other scenario.
        https://mathworld.wolfram.com/FrobeniusNorm.html

        As the matrices amongst scenarios might not match in shape,
        we conform all adjacency matrices to the common nodes of the base scenario
        and the other scenario being considered.

        Arguments:
            None

        Returns:
            None
        """
        self.frobenius = [
            np.linalg.norm(
                nx.to_numpy_array(
                    self.scenarios[0].trade_graph, nodelist=elligible_nodes
                )
                - nx.to_numpy_array(scenario.trade_graph, nodelist=elligible_nodes)
            )
            for scenario, elligible_nodes in zip(
                self.scenarios[1:], self.elligible_countries
            )
        ]

    def _compute_stationary_markov_distance(self) -> None:
        """
        Computes the 'Markov' distance between the base scenario graph
        and each other scenario.
        This is the Eucledean distance between stationary probability distribution
        vectors assuming a Markov random walk on the trade graphs.

        As the graph nodes amongst scenarios might not match, we conform all
        adjacency matrices to the common nodes of the base scenario
        and the other scenario being considered.

        Arguments:
            None

        Returns:
            None
        """
        subgraphs_with_elligible_nodes = [
            (
                nx.subgraph(self.scenarios[0].trade_graph, nbunch=elligible_nodes),
                nx.subgraph(scenario.trade_graph, nbunch=elligible_nodes),
            )
            for scenario, elligible_nodes in zip(
                self.scenarios[1:], self.elligible_countries
            )
        ]
        stationary_distribution_vectors = [
            [
                get_stationary_probability_vector(
                    get_right_stochastic_matrix(trade_matrix)
                )
                for trade_matrix in trade_matrix_pair
            ]
            for trade_matrix_pair in subgraphs_with_elligible_nodes
        ]
        self.markov = [
            np.linalg.norm(base_scenario_vec - vec)
            for (base_scenario_vec, vec) in stationary_distribution_vectors
        ]

    def _compute_entropy_rate_distance(self) -> None:
        """
        Computes the 'entropy rate' distance between the base scenario
        and each other scenario.
        This is a simple difference between entropy rates computed for each graph
        assuming a Markov random walk as the process.

        Arguments:
            None

        Returns:
            None
        """
        entropy_rates = [
            get_entropy_rate(scenario.trade_graph) for scenario in self.scenarios
        ]
        # compute difference from base scenario
        self.entropy_rate = [er - entropy_rates[0] for er in entropy_rates[1:]]

    def _format_distance_dataframe(self) -> None:
        """
        Creates a dataframe containing all computed graph difference  metrics.

        Arguments:
            None

        Returns:
            None
        """
        df = pd.DataFrame(
            zip(
                range(1, len(self.scenarios)),
                self.frobenius,
                self.markov,
                self.entropy_rate,
            ),
            columns=["Scenario ID", "Frobenius", "Markov", "Entropy rate"],
        )
        self.distance_df = df

    def print_distance_metrics(
        self, tablefmt="fancy_grid", file: str | None = None, **kwargs
    ) -> None:
        """
        Prints the graph distance metrics in a neat tabulated form.

        Arguments:
            tablefmt (str, optional): table format as expected by the tabulate package.
            file (str | None, optional): The file to which we print the result.
                If `None`, prints to standard output.
            **kwargs: any other keyworded arguments recognised by the tabulate package
                or pandas' `to_html` method if file is specified.

        Returns:
            None
        """
        df = self.distance_df.copy()
        df.set_index("Scenario ID", drop=True, inplace=True)
        if file:
            df.to_html(buf=file, **kwargs)
        else:
            print("***| Graph distance to the base scenario |***")
            print(df.to_markdown(tablefmt=tablefmt, **kwargs))

    def plot_distance_metrics(
        self,
        frobenius: str | None = None,
        figsize: tuple[float, float] | None = None,
        file_path: str | None = None,
        file_format="png",
        dpi=300,
        **kwargs,
    ) -> None:
        """
        Plots the distance metrics as a bar plot.

        Arguments:
            frobenius (str | None, optional): Flag controlling the behaviour of
                graph difference metrics.
                If frobenius == "relative" *all* metrics are normalised relative
                to the highest found value in each category; if "ignore" then
                frobenius will not be included in the the plot; if None, nothing
                special happens -- in this case the Frobenius metrics is very likely
                to completety overshadow other values in the plot.
            figsize (tuple[float, float] | None, optional): the composite figure
                size as expected by the matplotlib subplots routine.
            file_path (str | None, optional): Path to where the image file
                should be saved to. If `None` no file shall be produced.
            file_format (str, optional): File extension to use when
                saving plot to file.
            dpi (int, optional): DPI of the saved image file.
            **kwargs: any keyworded arguments recognised by seaborn barplot.

        Returns:
            None

        """
        df = self.distance_df.copy()
        match frobenius:
            case "relative":
                df[df.columns[1:]] = df[df.columns[1:]] / df[df.columns[1:]].max()
            case "ignore":
                df.drop(columns=["Frobenius"], inplace=True)
            case None:
                pass
            case _:
                print(
                    "Unrecognised option for plotting Frobenius, defaulting to 'relative'."
                )
                df[df.columns[1:]] = df[df.columns[1:]] / df[df.columns[1:]].max()
        df = df.melt(id_vars="Scenario ID", value_vars=df.columns[1:])
        _, ax = plt.subplots(figsize=figsize)
        sns.barplot(
            df,
            x="Scenario ID",
            y="value",
            hue="variable",
            **kwargs,
        )
        plt.ylabel("Distance")
        plt.title("Graph distance to the base scenario")
        if file_path:
            plt.savefig(
                f"{file_path}{os.sep}network_distance.{file_format}",
                dpi=dpi,
                bbox_inches="tight",
            )
        else:
            plt.show()

    def _compute_centrality(self) -> None:
        """
        Computes the in-degree and out-degree centrality for each node in each scenario.
        By in/out-degree centrality we understand sum of in/out-coming edge weights
        to/from a given node, divided by the total in/out-flow of the graph
        (sum of all in/out-coming edge weights in the graph).

        Arguments:
            None

        Returns:
            None
        """
        self.in_degree = [
            get_degree_centrality(scenario.trade_graph, out=False)
            for scenario in self.scenarios
        ]
        self.out_degree = [
            get_degree_centrality(scenario.trade_graph, out=True)
            for scenario in self.scenarios
        ]

    def _compute_global_centrality_metrics(self) -> None:
        """
        Computes global centrality metrics, i.e., for each scenario it finds
        the highest/lowst in/out-degree centrality nodes.

        Arguments:
            None

        Returns:
            None
        """
        centrality_metrics = []
        for idx, (in_d, out_d) in enumerate(zip(self.in_degree, self.out_degree)):
            in_min_max = get_dict_min_max(in_d)
            out_min_max = get_dict_min_max(out_d)
            centrality_metrics.append([idx, *in_min_max, *out_min_max])
        self.global_centrality_metrics = centrality_metrics

    def print_global_centrality_metrics(
        self, tablefmt="fancy_grid", file: str | None = None, **kwargs
    ) -> None:
        """
        Prints the global centrality metrics in a neat tabulated form.

        Arguments:
            tablefmt (str, optional): table format as expected by the tabulate package.
            file (str | None, optional): The file to which we print the result.
                If `None`, prints to standard output.
            **kwargs: any other keyworded arguments recognised by the tabulate package
                or pandas' `to_html` method if file is specified.

        Returns:
            None
        """
        df = pd.DataFrame(
            self.global_centrality_metrics,
            columns=[
                "Scenario\nID",
                "Smallest\nin-degree\ncountry",
                "Smallest\nin-degree\nvalue",
                "Largest\nin-degree\ncountry",
                "Largest\nin-degree\nvalue",
                "Smallest\nout-degree\ncountry",
                "Smallest\nout-degree\nvalue",
                "Largest\nout-degree\ncountry",
                "Largest\nout-degree\nvalue",
            ],
        )
        df = df.set_index("Scenario\nID", drop=True)
        if file:
            df.to_html(buf=file, **kwargs)
        else:
            print("***| Degree centrality metrics for each scenario |***")
            print(df.to_markdown(tablefmt=tablefmt, **kwargs))

    def _compute_community_centrality_metrics(self) -> None:
        """
        Computes local centrality metrics, i.e., for each scenario it finds
        the highest/lowst in/out-degree centrality nodes in each community.

        Arguments:
            None

        Returns:
            None
        """
        centrality_metrics = []
        for scenario_id, scenario in enumerate(self.scenarios):
            per_community_centrality_metrics = []
            for comm_id, community in enumerate(scenario.trade_communities):
                in_d = self.in_degree[scenario_id]
                out_d = self.out_degree[scenario_id]
                in_d = {k: v for k, v in in_d.items() if k in community}
                out_d = {k: v for k, v in out_d.items() if k in community}

                in_min_max = get_dict_min_max(in_d)
                out_min_max = get_dict_min_max(out_d)
                per_community_centrality_metrics.append(
                    [comm_id, *in_min_max, *out_min_max]
                )
            centrality_metrics.append(per_community_centrality_metrics)
        self.community_centrality_metrics = centrality_metrics

    def print_per_community_centrality_metrics(
        self, tablefmt="fancy_grid", file: str | None = None, **kwargs
    ) -> None:
        """
        Prints the local centrality metrics (per community) in a neat tabulated form.

        Arguments:
            tablefmt (str, optional): table format as expected by the tabulate package.
            file (str | None, optional): The file to which we print the result.
                If `None`, prints to standard output.
            **kwargs: any other keyworded arguments recognised by the tabulate package
                or pandas' `to_html` method if file is specified.

        Returns:
            None
        """
        for scenario_id, per_community_centrality_metrics in enumerate(
            self.community_centrality_metrics
        ):
            df = pd.DataFrame(
                per_community_centrality_metrics,
                columns=[
                    "Community\nID",
                    "Smallest\nin-degree\ncountry",
                    "Smallest\nin-degree\nvalue",
                    "Largest\nin-degree\ncountry",
                    "Largest\nin-degree\nvalue",
                    "Smallest\nout-degree\ncountry",
                    "Smallest\nout-degree\nvalue",
                    "Largest\nout-degree\ncountry",
                    "Largest\nout-degree\nvalue",
                ],
            )
            df = df.set_index("Community\nID", drop=True)
            if file:
                file.write(
                    f"<h3> Degree centrality metrics for the scenario with ID: {scenario_id}"
                )
                df.to_html(buf=file, **kwargs)
            else:
                print(
                    f"***| Degree centrality metrics for the scenario with ID: {scenario_id} |***"
                )
                print(df.to_markdown(tablefmt=tablefmt, **kwargs))

    def plot_centrality_maps(
        self,
        figsize: tuple[float, float] | None = None,
        shrink=1.0,
        file_path: str | None = None,
        file_format="png",
        dpi=300,
        **kwargs,
    ) -> None:
        """
        Plots world maps for each scenario, with each country coloured by their
        degree centrality in the trade graph.

        Arguments:
            figsize (tuple[float, float] | None, optional): the composite figure
                size as expected by the matplotlib subplots routine.
            shrink (float, optional): colour bar shrink parameter
            file_path (str | None, optional): Path to where the image file
                should be saved to. If `None` no file shall be produced.
            file_format (str, optional): File extension to use when
                saving plot to file.
            dpi (int, optional): DPI of the saved image file.
            **kwargs (optional): any additional keyworded arguments recognised
                by geopandas' plot function.

        Returns:
            None
        """
        _, axs = plt.subplots(
            2 * len(self.scenarios),
            1,
            sharex=True,
            tight_layout=True,
            figsize=(5, len(self.scenarios) * 5) if figsize is None else figsize,
        )
        # if there is only one scenario axs will be just an ax object
        # convert to a list to comply with other cases
        try:
            len(axs)
        except TypeError:
            axs = [axs]

        idx = 0
        for scenario, in_degree, out_degree in zip(
            self.scenarios, self.in_degree, self.out_degree
        ):
            plot_node_metric_map(
                axs[idx],
                scenario,
                in_degree,
                "in-degree",
                shrink=shrink,
                **kwargs,
            )
            plot_node_metric_map(
                axs[idx + 1],
                scenario,
                out_degree,
                "out-degree",
                shrink=shrink,
                **kwargs,
            )
            idx += 2
        if file_path:
            plt.savefig(
                f"{file_path}{os.sep}centrality_map.{file_format}",
                dpi=dpi,
                bbox_inches="tight",
            )
        else:
            plt.show()

    def _find_new_order(self, scenario: PyTradeShifts) -> list[set[str]]:
        """
        Computes the new order of communities such that the anchor countries'
        communities are always in the same position across scenarios.

        Arguments:
            scenario (PyTradeShifts): a PyTradeShifts object instance.

        Returns:
            list[set[str]]: List containing communities in the new order.
        """
        # make sure there are communities computed
        assert scenario.trade_communities is not None
        # find location of anchor countries in communities' list
        anchor_idx = {}
        for anchor in self.anchor_countries:
            for idx, community in enumerate(scenario.trade_communities):
                if anchor in community:
                    anchor_idx[anchor] = idx
                    break
        # make sure indices are unqiue, they wouldn't be if user passed
        # two or more countries from the same community as anchors
        assert len(anchor_idx.values()) == len(
            set(anchor_idx.values())
        ), "Two or more countries from the same community have been passed as anchores."
        # create new arrangement
        new_order = list(anchor_idx.values())
        # append remaining (i.e., un-anchored) community indices
        new_order.extend(
            set(range(len(scenario.trade_communities))).difference(new_order)
        )
        # make sure we've got that right
        assert len(new_order) == len(scenario.trade_communities)
        assert len(new_order) == len(set(new_order))
        # get communities in the new order
        return list(itemgetter(*new_order)(scenario.trade_communities))

    def _arrange_communities(self) -> None:
        """
        Orders communities in each scenario based on anchor countries such that
        the colours amongst community plots are more consistent.

        Arguments:
            None

        Returns:
            None
        """
        for scenario in self.scenarios:
            scenario.trade_communities = self._find_new_order(scenario)

    def _find_community_difference(self) -> None:
        """
        For each country and scenario, computes community similarity score
        (as Jaccard Index) in comparison with the base scenario communities.

        Arguments:
            None

        Returns:
            None
        """
        # initialise the similarity score dictionary
        jaccard_indices = {
            scenario_idx: {} for scenario_idx, _ in enumerate(self.scenarios[1:], 1)
        }
        # compute the scores
        for scenario_idx, scenario in enumerate(self.scenarios[1:], 1):
            for country in self.scenarios[0].trade_matrix.index:
                # this assumes that a country can be only in one community
                # i.e. there is no community overlap
                # find the community in which the country is the current scenario
                new_community = next(
                    filter(
                        lambda community: country in community,
                        scenario.trade_communities,
                    ),
                    None,
                )
                if new_community is None:
                    print(
                        f"Warning: {country} has no community in scenario {scenario_idx}."
                    )
                    continue
                else:
                    # we want to compare how the community changed from the perspective
                    # of the country so we need to exclude the country itself from
                    # the comparison
                    new_community = new_community - {country}

                # find the community in which the country is the base scenario
                original_community = next(
                    filter(
                        lambda community: country in community,
                        self.scenarios[0].trade_communities,
                    ),
                    None,
                )
                if original_community is None:
                    print(
                        f"Warning: {country} has no community in base scenario. Skipping."
                    )
                    continue
                else:
                    original_community = original_community - {country}

                jaccard_indices[scenario_idx][country] = jaccard_index(
                    new_community, original_community
                )
        self.jaccard_indices = jaccard_indices

    def plot_community_difference(
        self,
        similarity=False,
        figsize: tuple[float, float] | None = None,
        shrink=1.0,
        file_path: str | None = None,
        file_format="png",
        dpi=300,
        **kwargs,
    ):
        """
        Plots the world map for each scenario where each country's colour is the
        dissimilarity score with the base scenario of their communities.

        Arguments:
            similarity (bool, optional): whether to plot Jaccard index or distance.
                If True similarity (index) will be used, if False, distance = (1-index).
                Defualt is False.
            figsize (tuple[float, float] | None, optional): the composite figure
                size as expected by the matplotlib subplots routine.
            shrink (float, optional): colour bar shrink parameter
            file_path (str | None, optional): Path to where the image file
                should be saved to. If `None` no file shall be produced.
            file_format (str, optional): File extension to use when
                saving plot to file.
            dpi (int, optional): DPI of the saved image file.
            **kwargs (optional): any additional keyworded arguments recognised
                by geopandas plot function.

        Returns:
            None
        """
        assert len(self.scenarios) > 1
        _, axs = plt.subplots(
            len(self.scenarios) - 1,
            1,
            sharex=True,
            tight_layout=True,
            figsize=(
                (5, (len(self.scenarios) - 1) * 2.5) if figsize is None else figsize
            ),
        )
        # if there are only two scenarios axs will be just an ax object
        # convert to a list to comply with other cases
        try:
            len(axs)
        except TypeError:
            axs = [axs]
        for ax, (idx, scenario) in zip(axs, enumerate(self.scenarios[1:], 1)):
            plot_node_metric_map(
                ax,
                scenario,
                (
                    self.jaccard_indices[idx]
                    if similarity
                    else {k: 1 - v for k, v in self.jaccard_indices[idx].items()}
                ),
                metric_name="Jaccard similarity" if similarity else "Jaccard distance",
                shrink=shrink,
                **kwargs,
            )
        if file_path:
            plt.savefig(
                f"{file_path}{os.sep}community_diff.{file_format}",
                dpi=dpi,
                bbox_inches="tight",
            )
        else:
            plt.show()

    def plot_all_trade_communities(
        self,
        figsize: tuple[float, float] | None = None,
        file_path: str | None = None,
        file_format="png",
        dpi=300,
    ) -> None:
        """
        Plots the trade communities in each of the scenarios.

        Arguments:
            figsize (tuple[float, float] | None, optional): the composite figure
                size as expected by the matplotlib subplots routine.
            file_path (str | None, optional): Path to where the image file
                should be saved to. If `None` no file shall be produced.
            file_format (str, optional): File extension to use when
                saving plot to file.
            dpi (int, optional): DPI of the saved image file.

        Returns:
            None
        """
        _, axs = plt.subplots(
            len(self.scenarios),
            1,
            sharex=True,
            tight_layout=True,
            figsize=((5, len(self.scenarios) * 2.5) if figsize is None else figsize),
        )
        # if there is only one scenario axs will be just an ax object
        # convert to a list to comply with other cases
        try:
            len(axs)
        except TypeError:
            axs = [axs]
        for ax, scenario in zip(axs, self.scenarios):
            scenario._plot_trade_communities(ax)
        if file_path:
            plt.savefig(
                f"{file_path}{os.sep}trade_communities.{file_format}",
                dpi=dpi,
                bbox_inches="tight",
            )
        else:
            plt.show()

    def _compute_community_satisfaction(self) -> None:
        """
        Computes the community satisfaction index for each node.
        This index quantifies how much of a given country's import is satisfied
        by its community.
        The metric is taken from:
        Wang, X., Ma, L., Yan, S., Chen, X., & Growe, A. (2023).
        Trade for food security: The stability of global agricultural trade networks.
        Foods, 12(2), 271.
        https://www.mdpi.com/2304-8158/12/2/271, and
        Ji, Q., Zhang, H. Y., & Fan, Y. (2014).
        Identification of global oil trade patterns: An empirical research based
        on complex network theory.
        Energy Conversion and Management, 85, 856-865.
        https://www.sciencedirect.com/science/article/abs/pii/S0196890414000466.

        Arguments:
            None

        Returns:
            None
        """
        self.community_satisfaction = []
        for scenario in self.scenarios:
            # get total in-degree
            in_degrees = dict(scenario.trade_graph.in_degree(weight="weight"))
            for community in scenario.trade_communities:
                # get in-degree within the community only
                community_in_degrees = nx.subgraph(
                    scenario.trade_graph, nbunch=community
                ).in_degree(weight="weight")
                # replace in_degree with satisfaction index to save space
                for country, c_i_d in dict(community_in_degrees).items():
                    try:
                        in_degrees[country] = c_i_d / in_degrees[country]
                    except ZeroDivisionError:
                        # zero division means no import which is usually fine
                        # but if c_i_d != 0 at the same time then something
                        # is wrong
                        assert c_i_d == in_degrees[country]
                        continue
            self.community_satisfaction.append(in_degrees)

    def _compute_community_satisfaction_difference(self) -> None:
        """
        Computes the difference of the community satisfaction index between
        each scenario and the base scenario.

        Arguments:
            None

        Returns:
            None
        """
        self.community_satisfaction_difference = [
            {
                country: (
                    satisfaction - self.community_satisfaction[0][country]
                    if country in self.community_satisfaction[0]
                    else (
                        print(f"Warning: {country} not found in the base scenario.")
                        or np.nan
                    )
                )
                for country, satisfaction in community_satisfaction.items()
            }
            for community_satisfaction in self.community_satisfaction[1:]
        ]

    def plot_community_satisfaction(
        self,
        figsize: tuple[float, float] | None = None,
        shrink=1.0,
        file_path: str | None = None,
        file_format="png",
        dpi=300,
        **kwargs,
    ) -> None:
        """
        Plots the world map with countries coloured by their community
        satisfaction index.

        Arguments:
            figsize (tuple[float, float] | None, optional): The composite figure
                size as expected by the matplotlib subplots routine.
            shrink (float, optional): Colour bar shrink parameter.
            file_path (str | None, optional): Path to where the image file
                should be saved to. If `None` no file shall be produced.
            file_format (str, optional): File extension to use when
                saving plot to file.
            dpi (int, optional): DPI of the saved image file.
            **kwargs (optional): Any additional keyworded arguments recognised
                by geopandas plot function.

        Returns:
            None
        """
        _, axs = plt.subplots(
            len(self.scenarios),
            1,
            sharex=True,
            tight_layout=True,
            figsize=(5, len(self.scenarios) * 2.5) if figsize is None else figsize,
        )
        # if there is only one scenario axs will be just an ax object
        # convert to a list to comply with other cases
        try:
            len(axs)
        except TypeError:
            axs = [axs]
        for ax, (idx, scenario) in zip(axs, enumerate(self.scenarios)):
            plot_node_metric_map(
                ax,
                scenario,
                self.community_satisfaction[idx],
                "Community satisfaction",
                shrink=shrink,
                **kwargs,
            )
        if file_path:
            plt.savefig(
                f"{file_path}{os.sep}community_satisfaction.{file_format}",
                dpi=dpi,
                bbox_inches="tight",
            )
        else:
            plt.show()

    def plot_community_satisfaction_difference(
        self,
        figsize: tuple[float, float] | None = None,
        shrink=1.0,
        file_path: str | None = None,
        file_format="png",
        dpi=300,
        **kwargs,
    ) -> None:
        """
        Plots the world map with countries coloured by the difference of their
        community satisfaction index from the base scenario.

        Arguments:
            figsize (tuple[float, float] | None, optional): The composite figure
                size as expected by the matplotlib subplots routine.
            shrink (float, optional): Colour bar shrink parameter.
            file_path (str | None, optional): Path to where the image file
                should be saved to. If `None` no file shall be produced.
            file_format (str, optional): File extension to use when
                saving plot to file.
            dpi (int, optional): DPI of the saved image file.
            **kwargs (optional): Any additional keyworded arguments recognised
                by geopandas plot function.

        Returns:
            None
        """
        assert len(self.scenarios) > 1
        _, axs = plt.subplots(
            len(self.scenarios) - 1,
            1,
            sharex=True,
            tight_layout=True,
            figsize=(
                (5, (len(self.scenarios) - 1) * 2.5) if figsize is None else figsize
            ),
        )
        # if there are only two scenarios axs will be just an ax object
        # convert to a list to comply with other cases
        try:
            len(axs)
        except TypeError:
            axs = [axs]
        for ax, (idx, scenario) in zip(axs, enumerate(self.scenarios[1:])):
            plot_node_metric_map(
                ax,
                scenario,
                self.community_satisfaction_difference[idx],
                "Community satisfaction difference",
                shrink=shrink,
                **kwargs,
            )
        if file_path:
            plt.savefig(
                f"{file_path}{os.sep}community_satisfaction_diff.{file_format}",
                dpi=dpi,
                bbox_inches="tight",
            )
        else:
            plt.show()

    def _compute_efficiency(self) -> None:
        """
        Computes graph efficiency score for each scenario, based on:
        Bertagnolli, G., Gallotti, R., & De Domenico, M. (2021).
        Quantifying efficient information exchange in real network flows.
        Communications Physics, 4(1), 125.
        https://www.nature.com/articles/s42005-021-00612-5.

        Arguments:
            None

        Returns:
            None
        """
        self.efficiency = [
            get_graph_efficiency(scenario.trade_graph, self.normalisation)
            for scenario in self.scenarios
        ]

    def _compute_clustering_coefficient(self) -> None:
        """
        Computes the average clustering coefficient for each scenario.

        Arguments:
            None

        Returns:
            None
        """
        self.clustering = [
            nx.average_clustering(scenario.trade_graph, weight="weight")
            for scenario in self.scenarios
        ]

    def _compute_betweenness_centrality(self) -> None:
        """
        Computes the betweenness centrality for each scenario.
        Here, we consider the weight of the edge to be the inverse
        for the trade value as trade is more of a flow than a cost metric on a graph.

        Arguments:
            None

        Returns:
            None
        """
        self.betweenness = [
            np.mean(
                list(
                    nx.betweenness_centrality(
                        scenario.trade_graph, weight="1/weight"
                    ).values()
                )
            )
            for scenario in self.scenarios
        ]

    def _compute_within_community_degree(self) -> None:
        """
        Computes the z-score of node's degree within its community for each node
        in every scenario.
        Note: this metric ignores edge direction and weight.
        The metric is taken from:
        Guimerà, R., & Nunes Amaral, L. A. (2005).
        Functional cartography of complex metabolic networks.
        Nature, 433(7028), 895-900. doi:10.1038/nature03288
        https://www.nature.com/articles/nature03288

        Arguments:
            None

        Returns:
            None
        """
        self.zscores = []
        for scenario in self.scenarios:
            zscores = {}
            for community in scenario.trade_communities:
                community_subgraph = nx.subgraph(scenario.trade_graph, nbunch=community)
                community_subgraph = community_subgraph.to_undirected()
                subgraph_node_degrees = community_subgraph.degree()
                # we update the dict with countries and their degree z-scores
                # this assumes that each country can belong to a single community
                zscores |= dict(
                    zip(
                        [node for (node, _) in subgraph_node_degrees],
                        stats.zscore([degree for (_, degree) in subgraph_node_degrees]),
                    )
                )
            self.zscores.append(zscores)

    def _compute_participation(self) -> None:
        """
        Computes the participation coefficient for each node in every scenario.
        This metric is close to 1.0 if node's links are uniformly distributed
        among all the communities, and 0.0 if all its links are within its own
        community.
        Note: this metric ignores edge direction and weight.
        The metric is taken from:
        Guimerà, R., & Nunes Amaral, L. A. (2005).
        Functional cartography of complex metabolic networks.
        Nature, 433(7028), 895-900. doi:10.1038/nature03288
        https://www.nature.com/articles/nature03288

        Arguments:
            None

        Returns:
            None
        """
        self.participation = []
        for scenario in self.scenarios:
            undirected_trade_graph = scenario.trade_graph.to_undirected()
            total_degree = undirected_trade_graph.degree()
            coefficients = {
                # the score for a given node is 1 - 1/k^2 x sum_s (k_s)^2
                # where k is total node degree and the sum is over communities
                # where k_s is the number of edges from the node to community `s`
                country: 1.0
                - sum(
                    [
                        sum(
                            [  # this computes the number of edges from `country`
                                # to `community`
                                community_member in undirected_trade_graph[country]
                                for community_member in community
                            ]
                        )
                        ** 2
                        for community in scenario.trade_communities
                    ]  # we sum up the squares of the number of edges to each community
                )
                / (
                    total_degree[country] ** 2 if total_degree[country] != 0 else 1
                )  # Avoid division by zero
                for country in undirected_trade_graph  # for each country
            }
            self.participation.append(coefficients)

    def plot_roles(
        self,
        z_threshold=1.0,
        p_thresholds=[0.05, 0.3, 0.62, 0.75, 0.8],
        alpha=0.3,
        labels=True,
        fontsize=7,
        figsize: tuple[float, float] | None = None,
        file_path: str | None = None,
        file_format="png",
        countries_to_highlight: list[str] | None = None,
        offset_x: float = 0,
        offset_y: float = 0,
        dpi=300,
        **kwargs,
    ) -> None:
        """
        Generates a scatter plot where each point is a country, where Y values
        are the within community z-scores, and X values are participation coefficients.
        Both metrics are defined in:
        Guimerà, R., & Nunes Amaral, L. A. (2005).
        Functional cartography of complex metabolic networks.
        Nature, 433(7028), 895-900. doi:10.1038/nature03288
        https://www.nature.com/articles/nature03288

        Arguments:
            z_threshold (float, optional): value of z-score for a country to
                be considered a hub, default is 1.0.
            p_threshold (list[float], optional): list of participation
                coefficient threshold for a country to be provincial, connector,
                or kinless (and sub-categories, by default there are 7 different
                thresholds).
            alpha (float, optional): transparency level of the background colouring,
                showing the sectors defined by the above thresholds.
            labels (bool, optional): whether to plot labels in the plot.
            fontsize (int, optional): font size of the labels.
            figsize (tuple[float, float] | None, optional): the composite figure
                size as expected by the matplotlib subplots routine.
            shrink (float, optional): Colour bar shrink parameter.
            file_path (str | None, optional): Path to where the image file
                should be saved to. If `None` no file shall be produced.
            file_format (str, optional): File extension to use when
                saving plot to file.
            countries_to_highlight (list[str] | None, optional): List of countries
                to highlight in the plot.
            offset_x (float, optional): x-axis offset for labels of the highlighted countries.
            offset_y (float, optional): y-axis offset for labels of the highlighted countries.
            dpi (int, optional): DPI of the saved image file.
            **kwargs (optional): Any additional keyworded arguments recognised
                by geopandas plot function.

        Returns:
            None
        """
        _, axs = plt.subplots(
            len(self.scenarios),
            1,
            sharex=True,
            tight_layout=True,
            figsize=(7.5, len(self.scenarios) * 2.5) if figsize is None else figsize,
        )
        # if there is only one scenario axs will be just an ax object
        # convert to a list to comply with other cases
        try:
            len(axs)
        except TypeError:
            axs = [axs]
        for ax, (idx, scenario) in zip(axs, enumerate(self.scenarios)):
            # Make sure that countries are in the same order in both metrics by iterating over the countries
            # Saving them to a list
            countries_z = list(self.zscores[idx].keys())
            countries_p = list(self.participation[idx].keys())
            # Check if both lists contain the same countries
            assert set(countries_z) == set(countries_p)
            sorted_values_z = [self.zscores[idx][country] for country in countries_p]
            sorted_values_p = [
                self.participation[idx][country] for country in countries_p
            ]
            # Scatter plot
            ax.scatter(
                sorted_values_p,
                sorted_values_z,
                zorder=5,
                color="black",
                alpha=0.8,
                **kwargs,
            )
            ax.set_title(
                f"Country roles for {scenario.crop.lower()} with base year {scenario.base_year[1:]}"
                + (
                    f"\nin scenario: {scenario.scenario_name}"
                    if scenario.scenario_name is not None
                    else ""
                )
            )
            fill_sector_by_colour(
                ax, z_threshold, p_thresholds, alpha, labels, fontsize
            )
            if countries_to_highlight is not None:
                for country in countries_to_highlight:
                    try:
                        ax.annotate(
                            country,
                            xy=(
                                self.participation[idx][country],
                                self.zscores[idx][country],
                            ),
                            xytext=(
                                self.participation[idx][country] + offset_x,
                                self.zscores[idx][country] + offset_y,
                            ),
                            clip_on=True,
                            bbox={
                                "facecolor": "white",
                                "clip_on": True,
                                "alpha": 0.85,
                                "pad": 1,
                                "edgecolor": "none",
                            },
                            fontsize=fontsize,
                            zorder=10,
                        )
                        # highlight the country
                        ax.scatter(
                            self.participation[idx][country],
                            self.zscores[idx][country],
                            zorder=5,
                            color="red",
                            alpha=0.8,
                        )
                    except KeyError:
                        print(f"Warning: {country} not found in the data.")
            ax.set_xlabel("Participation coefficient (P)")
            ax.set_ylabel("Within community degree (z)")
            # Turn off the grid
            ax.grid(False)

        if file_path:
            plt.savefig(
                f"{file_path}{os.sep}country_roles.{file_format}",
                dpi=dpi,
                bbox_inches="tight",
            )
        else:
            plt.show()

    def _compute_node_stability(self) -> None:
        """
        Computes the node stability index for each node and scenario.
        The metric is taken from:
        Wang, X., Ma, L., Yan, S., Chen, X., & Growe, A. (2023).
        Trade for food security: The stability of global agricultural trade networks.
        Foods, 12(2), 271.
        https://www.mdpi.com/2304-8158/12/2/271, and
        Ji, Q., Zhang, H. Y., & Fan, Y. (2014).
        Identification of global oil trade patterns: An empirical research based
        on complex network theory.
        Energy Conversion and Management, 85, 856-865.
        https://www.sciencedirect.com/science/article/abs/pii/S0196890414000466.
        Note: the values here are somewhat arbitrary as the stability index
        and distance have unspecified units.

        Arguments:
            None

        Returns:
            None
        """
        self.node_stability = []
        # get the stability index, this is assumed to be provided by a file
        stability_index = get_stability_index(self.stability_index_file)
        # find the set of all countries across all scenarios
        # this way we can compute the distance matrix once
        # instead of for each scenario separately
        country_super_set = reduce(
            pd.Index.union, [sc.trade_matrix.index for sc in self.scenarios]
        )
        distance_matrix = get_distance_matrix(country_super_set, country_super_set)
        if distance_matrix is None:
            print("Node stability shall not be computed.")
            return
        for scenario, out_degree in zip(self.scenarios, self.out_degree):
            importer_stability_dict = {}
            for importer in scenario.trade_graph:
                importer_stability = 0
                # loop over countries with non-zero export
                for exporter, exporter_centrality in dict(
                    filter(lambda el: el[1] > 0, out_degree.items())
                ).items():
                    # one cannot export to oneself
                    if exporter == importer:
                        continue
                    if exporter not in stability_index:
                        print(f"{exporter} not found in stability index.")
                        continue
                    importer_stability += (
                        stability_index[exporter]
                        * exporter_centrality
                        * distance_matrix.loc[importer, exporter] ** -self.gamma
                    )
                importer_stability_dict[importer] = importer_stability
            self.node_stability.append(importer_stability_dict)

    def _compute_node_stability_difference(self) -> None:
        """
        Computes the node stability index difference between scenario and base scenario,
        for each node.
        The metric is taken from:
        Wang, X., Ma, L., Yan, S., Chen, X., & Growe, A. (2023).
        Trade for food security: The stability of global agricultural trade networks.
        Foods, 12(2), 271.
        https://www.mdpi.com/2304-8158/12/2/271, and
        Ji, Q., Zhang, H. Y., & Fan, Y. (2014).
        Identification of global oil trade patterns: An empirical research based
        on complex network theory.
        Energy Conversion and Management, 85, 856-865.
        https://www.sciencedirect.com/science/article/abs/pii/S0196890414000466.
        Note: the values here are somewhat arbitrary as the stability index
        and distance have unspecified units.
        Thus, this difference is computed as a relative difference: (y(i, n)-y(0, n))/y(0, n).
        Where y(0, n) is stability of node `n` in base scenario,
        and y(i, n) the stability of node `n` in scenario `i`.
        I.e., it measures the relative change from base scenario to another scenario.

        Arguments:
            None

        Returns:
            None
        """
        self.node_stability_difference = [
            {
                country: (
                    (stability - self.node_stability[0][country])
                    / self.node_stability[0][country]
                    if country in self.node_stability[0]
                    else (
                        print(f"Warning: {country} not found in the base scenario."),
                        np.nan,
                    )
                )
                for country, stability in node_stability.items()
            }
            for node_stability in self.node_stability[1:]
        ]

    def plot_node_stability(
        self,
        figsize: tuple[float, float] | None = None,
        shrink=1.0,
        file_path: str | None = None,
        file_format="png",
        dpi=300,
        **kwargs,
    ) -> None:
        """
        Plots the world map with countries coloured by the their stability index.
        The metric is taken from:
        Wang, X., Ma, L., Yan, S., Chen, X., & Growe, A. (2023).
        Trade for food security: The stability of global agricultural trade networks.
        Foods, 12(2), 271.
        https://www.mdpi.com/2304-8158/12/2/271, and
        Ji, Q., Zhang, H. Y., & Fan, Y. (2014).
        Identification of global oil trade patterns: An empirical research based
        on complex network theory.
        Energy Conversion and Management, 85, 856-865.
        https://www.sciencedirect.com/science/article/abs/pii/S0196890414000466.
        Note: the values here are somewhat arbitrary as the stability index
        and distance have unspecified units.
        Plotting the relative difference might be more useful.
        See: Postprocessing.plot_node_stability_difference.

        Arguments:
            figsize (tuple[float, float] | None, optional): the composite figure
                size as expected by the matplotlib subplots routine.
            shrink (float, optional): Colour bar shrink parameter.
            file_path (str | None, optional): Path to where the image file
                should be saved to. If `None` no file shall be produced.
            file_format (str, optional): File extension to use when
                saving plot to file.
            dpi (int, optional): DPI of the saved image file.
            **kwargs (optional): Any additional keyworded arguments recognised
                by geopandas plot function.

        Returns:
            None
        """
        _, axs = plt.subplots(
            len(self.scenarios),
            1,
            sharex=True,
            tight_layout=True,
            figsize=(5, len(self.scenarios) * 2.5) if figsize is None else figsize,
        )
        # if there is only one scenario axs will be just an ax object
        # convert to a list to comply with other cases
        try:
            len(axs)
        except TypeError:
            axs = [axs]
        for ax, (idx, scenario) in zip(axs, enumerate(self.scenarios)):
            plot_node_metric_map(
                ax,
                scenario,
                self.node_stability[idx],
                "Node stability",
                shrink=shrink,
                **kwargs,
            )
        if file_path:
            plt.savefig(
                f"{file_path}{os.sep}node_stability.{file_format}",
                dpi=dpi,
                bbox_inches="tight",
            )
        else:
            plt.show()

    def plot_node_stability_difference(
        self,
        figsize: tuple[float, float] | None = None,
        shrink=1.0,
        file_path: str | None = None,
        file_format="png",
        dpi=300,
        **kwargs,
    ) -> None:
        """
        Plots the world map with countries coloured by the difference between
        each scenario and the base scenario of their stability index.
        The metric is taken from:
        Wang, X., Ma, L., Yan, S., Chen, X., & Growe, A. (2023).
        Trade for food security: The stability of global agricultural trade networks.
        Foods, 12(2), 271.
        https://www.mdpi.com/2304-8158/12/2/271, and
        Ji, Q., Zhang, H. Y., & Fan, Y. (2014).
        Identification of global oil trade patterns: An empirical research based
        on complex network theory.
        Energy Conversion and Management, 85, 856-865.
        https://www.sciencedirect.com/science/article/abs/pii/S0196890414000466.
        Note: the values here are somewhat arbitrary as the stability index
        and distance have unspecified units.
        Thus, this difference is computed as a relative difference: (y(i, n)-y(0, n))/y(0, n).
        Where y(0, n) is stability of node `n` in base scenario,
        and y(i, n) the stability of node `n` in scenario `i`.
        I.e., it measures the relative change from base scenario to another scenario.

        Arguments:
            figsize (tuple[float, float] | None, optional): The composite figure
                size as expected by the matplotlib subplots routine.
            shrink (float, optional): Colour bar shrink parameter.
            file_path (str | None, optional): Path to where the image file
                should be saved to. If `None` no file shall be produced.
            file_format (str, optional): File extension to use when
                saving plot to file.
            dpi (int, optional): DPI of the saved image file.
            **kwargs (optional): Any additional keyworded arguments recognised
                by geopandas plot function.

        Returns:
            None
        """
        assert len(self.scenarios) > 1
        _, axs = plt.subplots(
            len(self.scenarios) - 1,
            1,
            sharex=True,
            tight_layout=True,
            figsize=(
                (5, (len(self.scenarios) - 1) * 2.5) if figsize is None else figsize
            ),
        )
        # if there are only two scenarios axs will be just an ax object
        # convert to a list to comply with other cases
        try:
            len(axs)
        except TypeError:
            axs = [axs]
        for ax, (idx, scenario) in zip(axs, enumerate(self.scenarios[1:])):
            plot_node_metric_map(
                ax,
                scenario,
                self.node_stability_difference[idx],
                "Node stability relative difference",
                shrink=shrink,
                **kwargs,
            )
        if file_path:
            plt.savefig(
                f"{file_path}{os.sep}node_stability_diff.{file_format}",
                dpi=dpi,
                bbox_inches="tight",
            )
        else:
            plt.show()

    def _compute_network_stability(self) -> None:
        """
        Computes the network stability based on the node stability index.
        It is the sum of node stabilities weighed by the in-degree centrality.
        The metric is taken from:
        Wang, X., Ma, L., Yan, S., Chen, X., & Growe, A. (2023).
        Trade for food security: The stability of global agricultural trade networks.
        Foods, 12(2), 271.
        https://www.mdpi.com/2304-8158/12/2/271, and
        Ji, Q., Zhang, H. Y., & Fan, Y. (2014).
        Identification of global oil trade patterns: An empirical research based
        on complex network theory.
        Energy Conversion and Management, 85, 856-865.
        https://www.sciencedirect.com/science/article/abs/pii/S0196890414000466.
        Note: the values here are somewhat arbitrary as the stability index
        and distance used in the node stability index have unspecified units.

        Arguments:
            None

        Returns:
            None
        """
        #  node stability can fail to compute so we need to handle such a case
        if not self.node_stability:
            self.network_stability = None
            return
        self.network_stability = [
            sum(
                [
                    importer_centrality * stability[importer]
                    for importer, importer_centrality in in_degree.items()
                ]
            )
            for in_degree, stability in zip(self.in_degree, self.node_stability)
        ]

    def _compute_entropic_out_degree(self) -> None:
        """
        Compute the entropic out-degree for each node, scenario.
        This is a generalisation of the concept introduced here:
        Bompard, E., Napoli, R., & Xue, F. (2009).
        Analysis of structural vulnerabilities in power transmission grids.
        International Journal of Critical Infrastructure Protection, 2(1-2), 5-12.
        https://www.sciencedirect.com/science/article/abs/pii/S1874548209000031.
        This metric uses the idea of entropy to calculnode staate an importance of a node.

        Arguments:
            None

        Returns:
            None
        """
        self.entropic_out_degree = [
            get_entropic_degree(scenario.trade_graph, out=True)
            for scenario in self.scenarios
        ]

    def _compute_percolation_threshold(self) -> None:
        """
        Computes percolation threshold (or the network collapse threshold) using
        the theory developed in:
        Restrepo, J. G., Ott, E., & Hunt, B. R. (2008).
        Weighted percolation on directed networks.
        Physical review letters, 100(5), 058701.
        https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.100.058701.
        The idea is that we represent the attack strategy by a vector `p`,
        then given an adjacency matrix A, the largest eigenvalue of the matrix A(1-p)
        is 1 at the critical percolation point. When this eigenvalue is than 1
        the network disintegrates.
        We consider three attack vectors here: random, export-based and entropy-based.
        In random we remove nodes randomly, in export-based in the order of highest
        to lowest out-degree centrality, and entropy-based in the order of highest
        to lowest entropic degree of the nodes.

        Arguments:
            None

        Returns:
            None
        """
        # initialise list of dicts
        self.percolation = [{} for _ in self.scenarios]
        for idx, (scenario, out_degree, entropic_out_degree) in enumerate(
            zip(self.scenarios, self.out_degree, self.entropic_out_degree)
        ):
            adj = nx.to_numpy_array(scenario.trade_graph)
            adj[adj != 0] = 1  # weights don't matter here
            # export based attack
            self.percolation[idx]["export"] = get_percolation_threshold(
                adj, out_degree.values()
            )
            # entropic export based attack
            self.percolation[idx]["entropic"] = get_percolation_threshold(
                adj, entropic_out_degree.values()
            )
            # random attack,
            # it needs special treatment, i.e., averaging over multiple realisations
            # there is probably some clever way of gradually increasing
            # a uniform p_i vector but this is easier and fast enough
            random_attacks = [
                get_percolation_threshold(adj, np.random.random(len(adj)))
                for _ in range(self.random_attack_sample_size)
            ]
            random_attacks_df = pd.concat(
                [
                    pd.DataFrame(
                        np.vstack((removed_nodes_count, eigenvalues)).T,
                        columns=["removed_nodes", "eigenvalue"],
                    )
                    for _, removed_nodes_count, eigenvalues in random_attacks
                ]
            )
            thresholds_vector = [thr for thr, _, _ in random_attacks]
            self.percolation[idx]["random"] = (
                np.mean(thresholds_vector),
                stats.sem(thresholds_vector),
                random_attacks_df,
            )

    def plot_attack_resilience(
        self,
        exclude_scenarios=[],
        exclude_attacks=[],
        sigma=2.0,
        figsize: tuple[float, float] | None = None,
        file_path: str | None = None,
        file_format="png",
        dpi=300,
    ) -> None:
        """
        Plots the attack resiliencie for each scenario and attack strategy.

        Arguments:
            exclude_scenarios (list, optional): list of scenario IDs (int) to
                not be plotted.
            exclude_attacks (list, optional): list of attack strategies (str) to
                not be plotted. Possible attack strategies are:
                `export`, `entropic`, `random`.
            sigma (float, optional): standard error of the mean scale factor. This
                is used for error bars in the random attack strategy.
                sigma=1.0 -> ~68% confidence interval, 2.0->95% CI, etc.
            figsize (tuple[float, float] | None, optional): the composite figure
                size as expected by the matplotlib subplots routine.
            file_path (str | None, optional): Path to where the image file
                should be saved to. If `None` no file shall be produced.
            file_format (str, optional): File extension to use when
                saving plot to file.
            dpi (int, optional): DPI of the saved image file.


        Returns:
            None
        """
        # this line shows the critical value
        _, ax = plt.subplots(figsize=figsize, tight_layout=True)
        ax.axhline(
            1,
            color="black",
            linestyle="dashed",
            label="network collapse; \n[ID, attack, threshold]:",
        )
        num_scenarios = max(2, len(self.scenarios))
        alpha_steps = 0.9 / num_scenarios
        colors = [
            "#6c7075",
            "#e06234",
            "#F0B323",
            "#3D87CB",
        ]
        for idx, x in enumerate(self.percolation):
            if idx in exclude_scenarios:
                continue
            alpha = 1
            if "export" not in exclude_attacks:
                threshold, removed_nodes, eigenvalues = x["export"]
                ax.plot(
                    removed_nodes,
                    eigenvalues,
                    "-",
                    label=f"{idx}, export-weighted, {threshold}",
                    color=colors[idx],
                    alpha=alpha,
                )
                alpha -= alpha_steps
            if "entropic" not in exclude_attacks:
                threshold, removed_nodes, eigenvalues = x["entropic"]
                ax.plot(
                    removed_nodes,
                    eigenvalues,
                    "-",
                    label=f"{idx}, entropic, {threshold}",
                    color=colors[idx],
                    alpha=alpha,
                )
                alpha -= alpha_steps
            if "random" not in exclude_attacks:
                threshold, threshold_sem, random_attacks_df = x["random"]
                sb.lineplot(
                    random_attacks_df,
                    x="removed_nodes",
                    y="eigenvalue",
                    errorbar=("se", sigma),
                    label=f"{idx}, random, {threshold:.2g} +/- {sigma*threshold_sem:.2g}",
                    ax=ax,
                    color=colors[idx],
                    alpha=alpha,
                )
        ax.set_xlabel("# of removed nodes.")
        ax.set_ylabel("Max adj. eigenval. post node removal.")
        ax.legend()
        if file_path:
            plt.savefig(
                f"{file_path}{os.sep}attack_resilience.{file_format}",
                dpi=dpi,
                bbox_inches="tight",
            )
        else:
            plt.show()

    def print_network_metrics(
        self, wide=True, tablefmt="fancy_grid", file: str | None = None, **kwargs
    ) -> None:
        """
        Prints general graph metrics in a neat tabulated form.

        Arguments:
            wide (bool, optional): Whether print the table in long or wide format.
            tablefmt (str, optional): table format as expected by the tabulate package.
            file (str | None, optional): The file to which we print the result.
                If `None`, prints to standard output.
            **kwargs: any other keyworded arguments recognised by the tabulate package
                or pandas' `to_html` method if file is specified.

        Returns:
            None
        """
        metrics = [
            *[
                (idx, "Efficiency", efficiency)
                for idx, efficiency in enumerate(self.efficiency)
            ],
            *[
                (idx, "Clustering", clustering)
                for idx, clustering in enumerate(self.clustering)
            ],
            *[
                (idx, "Betweenness", betweenness)
                for idx, betweenness in enumerate(self.betweenness)
            ],
        ]
        if self.network_stability is not None:
            metrics.extend(
                [
                    (idx, "Stability", stability)
                    for idx, stability in enumerate(self.network_stability)
                ],
            )
        metrics.extend(
            [
                (
                    idx,
                    f"{attack.title()}-attack\nthreshold",
                    threshold if attack != "random" else f"{threshold:.2g} +/- {_:.2g}",
                )
                for idx, scenario in enumerate(self.percolation)
                for attack, (threshold, _, __) in scenario.items()
            ]
        )
        metrics = pd.DataFrame(metrics, columns=["Scenario ID", "Metric", "Value"])
        if wide:
            metrics = metrics.pivot(
                index="Scenario ID", columns="Metric", values="Value"
            ).rename_axis(None, axis=1)
        if file:
            metrics.to_html(buf=file, **kwargs)
        else:
            print(metrics.to_markdown(tablefmt=tablefmt, **kwargs))

    def _plot_all_to_file(self, figures_folder) -> None:
        """
        A helper function generating all the plots used for the report.

        Arguments:
            figures_folder (str): Path to where the plots should be placed.

        Returns:
            None
        """
        self.plot_all_trade_communities(file_path=figures_folder)
        plt.close()  # this is to prevent cross-contamination of plots
        # there is no point in this plot if there's only base scenario
        print("Plotted trade communities")
        if len(self.scenarios) > 1:
            self.plot_community_difference(file_path=figures_folder)
            plt.close()
            print("Plotted difference in trade communities")
        # there is no point in this plot if there's only base scenario and
        # one other
        if len(self.scenarios) > 2:
            self.plot_distance_metrics(
                file_path=figures_folder, frobenius="relative", figsize=(5, 2.5)
            )
            plt.close()
            print("Plotted distance metrics")
        self.plot_centrality_maps(file_path=figures_folder)
        plt.close()
        print("Plotted centrality map")
        self.plot_community_satisfaction(file_path=figures_folder)
        plt.close()
        print("Plotted community satisfaction index")
        if len(self.scenarios) > 1:
            self.plot_community_satisfaction_difference(file_path=figures_folder)
            plt.close()
            print("Plotted difference in community satisfaction")
        self.plot_node_stability(file_path=figures_folder)
        plt.close()
        print("Plotted node stability map")
        if len(self.scenarios) > 1:
            self.plot_node_stability_difference(file_path=figures_folder)
            plt.close()
            print("Plotted difference in node stability")
        self.plot_attack_resilience(file_path=figures_folder)
        plt.close()
        print("Plotted attack resilience")
        self.plot_roles(
            c="black",
            sector_labels=[
                "Ultra peripheral non-hub",
                "Provincial hub",
                "Peripheral non-hub",
                "Connector hub",
                "Connector non-hub",
                "Kinless hub",
                "Kinless non-hub",
            ],
            file_path=figures_folder,
        )
        plt.close()
        print("Plotted country roles")

    def _write_to_report_file(self, report_file_path, time_now, utc_label) -> None:
        """
        A helper function generating all the HTML tags and data tables in the
        index.html file.

        Arguments:
            report_file_path (str): Path to the index.html file.
            time_now (datetime): A datetime object, presumably current date and time.
            utc_label (str): String to be added to the report signifying the timezone.
                Presumably this just says "UTC" when the utc flag is used in the
                report function; however, technically any string can be passed.

        Returns:
            None
        """
        with open(report_file_path, "w") as report_file:
            report_file.write(
                """<!DOCTYPE html> <html> <style> table {width: 50%;} img {width: 50%;} </style> <body>"""
            )
            report_file.write("<p><center>")
            report_file.write(
                f"""<h1> PyTradeShifts Report {time_now.replace("_", " ")} {utc_label} </h1> <br>"""
            )
            report_file.write("<h2> Trade communities </h2>")
            report_file.write("""<img src="figs/trade_communities.png" >""")
            if len(self.scenarios) > 1:
                report_file.write("<h2> Difference in trade communities </h2>")
                report_file.write("""<img src="figs/community_diff.png">""")
            report_file.write("<h2> Community satisfaction </h2>")
            report_file.write("""<img src="figs/community_satisfaction.png">""")
            if len(self.scenarios) > 1:
                report_file.write("<h2> Difference in community satisfaction </h2>")
                report_file.write(
                    """<img src="figs/community_satisfaction_diff.png" >"""
                )
            if len(self.scenarios) > 1:
                report_file.write(
                    "<h2> Graph structural difference to base scenario (ID=0) </h2>"
                )
                self.print_distance_metrics(file=report_file, justify="center")
            if len(self.scenarios) > 2:
                report_file.write("""<img src="figs/network_distance.png" >""")
            report_file.write("<h2> Centrality metrics by scenario </h2>")
            self.print_global_centrality_metrics(file=report_file, justify="center")
            report_file.write("<h2> Centrality metrics by community </h2>")
            self.print_per_community_centrality_metrics(
                file=report_file, justify="center"
            )
            report_file.write("<h2> Centrality map </h2>")
            report_file.write("""<img src="figs/centrality_map.png" >""")
            report_file.write("<h2> Stability map </h2>")
            report_file.write("""<img src="figs/node_stability.png" >""")
            if len(self.scenarios) > 1:
                report_file.write("<h2> Difference in stability </h2>")
                report_file.write("""<img src="figs/node_stability_diff.png" >""")
            report_file.write("<h2> General network characteristics </h2>")
            self.print_network_metrics(file=report_file, justify="center")
            report_file.write("<h2> Attack resilience </h2>")
            report_file.write("""<img src="figs/attack_resilience.png" >""")
            report_file.write("<h2> Country roles </h2>")
            report_file.write("""<img src="figs/country_roles.png" >""")
            report_file.write("</center></p>")
            report_file.write("</body> </html>")

    def report(self, path=f"results{os.sep}reports", utc=True) -> None:
        """
        Generates all results available in the postprocessing suite and formats
        them into a readable report in the specified path.
        Name of the report is date and time at the moment of calling the function,
        time is UTC or local (controlled by a function parameter).
        Note: this function does no allow for much personalisation of how the
        results are presented. The idea of this function is to be run ``as is''.
        If a custom solution is required use something else such as a jupyter notebook.
        An example of such a notebook is provided at: scripts/scenario_comparison.ipynb.

        Arguments:
            path (str, optional): The localtion where the report is to be put.
                This function will generate all necessary parent directories.
            utc (bool, optional): Whether to use UTC or local time in the report.

        Returns:
            None
        """
        time_now = datetime.now(timezone.utc) if utc else datetime.now()
        time_now = time_now.strftime("%Y-%m-%d_%H:%M:%S")
        utc_label = "UTC" if utc else ""
        report_folder = f"{path}{os.sep}report_{time_now}"
        Path(report_folder).mkdir(parents=True, exist_ok=True)
        figures_folder = f"{report_folder}{os.sep}figs"
        Path(figures_folder).mkdir(parents=True, exist_ok=True)
        report_file_path = f"{report_folder}{os.sep}index.html"
        print("Starting plotting procedures...")
        self._plot_all_to_file(figures_folder)
        print("Starting printing procedures...")
        self._write_to_report_file(report_file_path, time_now, utc_label)
        print("Fin.")
