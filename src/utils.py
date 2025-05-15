from typing import Iterable, Any
import numpy as np
import geopandas as gpd
import pandas as pd
import country_converter as coco
import os
from itertools import groupby
import networkx as nx
from matplotlib.axes import Axes
from geopy.distance import geodesic
from scipy.spatial.distance import squareform, pdist


def plot_winkel_tripel_map(ax: Axes):
    """
    Helper function to plot a Winkel Tripel map with a border.
    """
    border_geojson = gpd.read_file(
        "https://raw.githubusercontent.com/ALLFED/ALLFED-map-border/main/border.geojson"
    )
    border_geojson.plot(ax=ax, edgecolor="black", linewidth=0.1, facecolor="none")
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])


def prepare_centroids(trade_matrix_index: pd.Index) -> pd.DataFrame:
    """
    Prepares a DataFrame containing coordinates of centroids for each region.
    A centroid is the geometric centre of a region
    (also the centre of mass assuming uniform mass distribution).
    This method is used internally in the PyTradeShifts class for the
    computation of the distance matrix between regions.

    Arguments:
        trade_matrix_index (pd.Index): The index of the trade matrix.

    Returns:
        pd.DataFrame: Data frame with columns ["longitude", "latitude", "name"]
            containing centroid coordinates for each region.
    """
    # https://github.com/gavinr/world-countries-centroids
    centroids_a = pd.read_csv(
        "."
        + os.sep
        + "data"
        + os.sep
        + "geospatial_references"
        + os.sep
        + "country_centroid_locations.csv"
    )
    # convert whatever names data sets have to one format so that we can
    # merge all data convieniently later
    centroids_a["name"] = [
        str(c) for c in coco.convert(centroids_a["COUNTRY"], to="name_short")
    ]
    # we need this for Taiwan
    centroids_b = gpd.read_file(
        "."
        + os.sep
        + "data"
        + os.sep
        + "geospatial_references"
        + os.sep
        + "ne_110m_admin_0_countries"
    )
    centroids_b["name"] = [
        str(c) for c in coco.convert(centroids_b["NAME_LONG"], to="name_short")
    ]
    # we use natural earth data to compute centroids
    # https://gis.stackexchange.com/questions/372564/userwarning-when-trying-to-get-centroid-from-a-polygon-geopandas
    centroids_b["centroid"] = centroids_b.to_crs("+proj=cea").centroid.to_crs(
        centroids_b.crs
    )
    centroids_b["lon"] = centroids_b["centroid"].x
    centroids_b["lat"] = centroids_b["centroid"].y
    # filter out unneeded columns
    centroids_b = centroids_b[["lon", "lat", "name"]]
    # need to add some regions manually; values taken from Google maps
    centroids_b = pd.concat(
        [
            centroids_b,
            pd.DataFrame(
                [
                    [113.54474590904003, 22.198228394900145, "Macau"],
                    [114.17506394778538, 22.32788145032773, "Hong Kong"],
                ],
                columns=["lon", "lat", "name"],
            ),
        ]
    )
    # merge the centroid data sets
    centroids = centroids_a.merge(centroids_b, how="outer", on="name")
    # filter out the regions that aren't in the trade matrix or in centroids
    trade_matrix_corrected_index = [
        str(c) for c in coco.convert(trade_matrix_index, to="name_short")
    ]
    centroids = (
        centroids.merge(
            pd.DataFrame(
                trade_matrix_corrected_index,
                columns=["name"],
            ),
            how="inner",
            on="name",
        )
        .sort_values("name")
        .reset_index(drop=True)
    )
    # fill missing values from data set 'a' with values from data set 'b'
    centroids.loc[:, "longitude"].fillna(centroids["lon"], inplace=True)
    centroids.loc[:, "latitude"].fillna(centroids["lat"], inplace=True)
    # filter out unneeded columns and remove duplicates
    centroids = centroids[["longitude", "latitude", "name"]]
    centroids = centroids.drop_duplicates()
    trade_matrix_index_restoration_dict = {
        c: cc
        for c, cc in zip(trade_matrix_corrected_index, trade_matrix_index)
        if c in centroids.index
    }
    centroids["name"].replace(trade_matrix_index_restoration_dict, inplace=True)
    centroids.sort_values("name", inplace=True)
    return centroids


def all_equal(iterable: Iterable):
    """
    Checks if all the elements are equal to each other.
    source: https://docs.python.org/3/library/itertools.html#itertools-recipes

    Arguments:
        iterable (Iterable): a list-like object containing elements to compare.

    Returns:
        bool: True if all elements are equal, False otherwise.
    """
    g = groupby(iterable)
    return next(g, True) and not next(g, False)


def jaccard_index(iterable_A: Iterable, iterable_B: Iterable) -> float:
    """
    Computes the Jaccard similarity beween two iterables.
    https://en.wikipedia.org/wiki/Jaccard_index
    Note: the iterables shall be cast to sets.

    Arguments:
        iterable_A (Iterable): a list-like object
        iterable_B (Iterable): a list-like object

    Returns:
        float: The Jaccard index (similarity) of iterable_A and iterable_B
    """
    A = set(iterable_A)
    B = set(iterable_B)
    return len(A.intersection(B)) / len(A.union(B))


def get_degree_centrality(graph: nx.DiGraph, out=False) -> dict:
    """
    Provides degree centrality for a directed, weighted graph.
    Assumes the weight attribute to be `weight`.

    Arguments:
        graph (nx.DiGraph): Graph on which the degrees are computed.
        out (bool, optional): Whether to compute out- or in-degree.

    Returns:
        dict: The mapping of node -> degree centrality.
    """
    if out:
        degrees = list(graph.out_degree(weight="weight"))
    else:
        degrees = list(graph.in_degree(weight="weight"))
    total_degrees = sum(map(lambda t: t[1], degrees))
    return dict(map(lambda t: (t[0], t[1] / total_degrees), degrees))


def get_entropic_degree(graph: nx.DiGraph, out=True) -> dict:
    """
    Compute the entropic in-/out-degree for each node in a directed, weighted graph.
    This is a generalisation of the concept introduced here:
    Bompard, E., Napoli, R., & Xue, F. (2009).
    Analysis of structural vulnerabilities in power transmission grids.
    International Journal of Critical Infrastructure Protection, 2(1-2), 5-12.
    https://www.sciencedirect.com/science/article/abs/pii/S1874548209000031.
    This metric uses the idea of entropy to calculate an importance of a node.

    Arguments:
        graph (nx.DiGraph): The directed, weighted graph whose nodes are to be evaluated.
        out (bool, optional): whether to compute the out- or in-degree. Default is out-degree.

    Returns:
        dict: The mapping of node -> entropic degree.
    """
    if out:
        degree = graph.out_degree(weight="weight")
    else:
        degree = graph.in_degree(weight="weight")
    entropic_degree = {}
    for n in graph:
        if degree[n] == 0:
            entropic_degree[n] = 0
            continue
        # p_ij is the normalised edge weight between the nodes i,j
        # by the sum of edge weights at node j
        # here we only consider outward pointing edges
        p = np.fromiter(
            map(
                lambda x: x[2]["weight"] / degree[n],
                graph.out_edges(n, data=True) if out else graph.in_edges(n, data=True),
            ),
            dtype=float,
            count=len(graph[n]),
        )
        # 0 * log(0) = 0 in information theory
        entropic_degree[n] = (1 - np.sum(p * np.nan_to_num(np.log(p)))) * degree[n]
    return entropic_degree


def prepare_world() -> gpd.GeoDataFrame:
    """
    Prepares the geospatial Natural Earth (NE) data (to be presumebly used in plotting).

    Arguments:
        None

    Returns:
        gpd.GeoDataFrame: NE data projected to Winke Tripel and with converted country names.
    """
    # get the world map
    world = gpd.read_file(
        "."
        + os.sep
        + "data"
        + os.sep
        + "geospatial_references"
        + os.sep
        + "ne_110m_admin_0_countries"
        + os.sep
        + "ne_110m_admin_0_countries.shp"
    )
    # Change projection to Winkel Tripel
    world = world.to_crs("+proj=wintri")

    cc = coco.CountryConverter()
    world["names_short"] = cc.pandas_convert(pd.Series(world["ADMIN"]), to="name_short")
    return world


def plot_node_metric_map(
    ax: Axes, scenario, metric: dict, metric_name: str, shrink=1.0, **kwargs
) -> None:
    """
    Plots world map with countries coloured by the specified metric.

    Arguments:
        ax (Axes): axis on which to plot.
        scenario (PyTradeShifts): a PyTradeShifts instance.
        metric (dict): dictionary containing the mapping: country->value
        metric_name (str): the name of the metric
        shrink (float, optional): colour bar shrink parameter
        **kwargs (optional): any additional keyworded arguments recognised
            by geopandas plot function.

    Returns:
        None
    """
    assert scenario.trade_communities is not None
    world = prepare_world()

    # Join the country_community dictionary to the world dataframe
    world[metric_name] = world["names_short"].map(metric)

    world.plot(
        ax=ax,
        column=metric_name,
        missing_kwds={"color": "lightgrey"},
        legend=True,
        legend_kwds={
            "label": metric_name,
            "shrink": shrink,
        },
        **kwargs,
    )

    plot_winkel_tripel_map(ax)

    # Add a title with self.scenario_name if applicable
    ax.set_title(
        (
            f"{metric_name.replace(' [%]', '').replace(' [t]', '')} for "
            f"{scenario.crop.lower()} with base year {scenario.base_year[1:]}"
            + (
                f"\nin scenario: {scenario.scenario_name}"
                if scenario.scenario_name is not None
                else ""
            )
        )
    )


def get_right_stochastic_matrix(trade_graph: nx.Graph) -> np.ndarray:
    """
    Convert graph's adjacency matrix to a right stochastic matrix (RSM).
    https://en.wikipedia.org/wiki/Stochastic_matrix

    Arguments:
        trade_graph (networkx.Graph): the graph object

    Returns:
        numpy.ndarray: an array representing the RSM.
    """
    # extract the adjaceny matrix from the graph
    right_stochastic_matrix = nx.to_pandas_adjacency(trade_graph)
    # normalise the matrix such that each row sums up to 1
    right_stochastic_matrix = right_stochastic_matrix.div(
        right_stochastic_matrix.sum(axis=1), axis=0
    )
    right_stochastic_matrix.fillna(0, inplace=True)
    return right_stochastic_matrix.values


def get_stationary_probability_vector(
    right_stochastic_matrix: np.ndarray,
) -> np.ndarray:
    """
    Find the stationary probability distribution for a given
    right stochastic matrix (RMS).

    Arguments:
        right_stochastic_matrix (numpy.ndarray): an array represnting the RMS.

    Returns:
        numpy.ndarray: a vector representing the stationary distribution.
    """
    # get eigenvalues and (row) eigenvectors
    eigenvalues, eigenvectors = np.linalg.eig(right_stochastic_matrix.T)
    # we want eigenvector associated with eigenvalue = 1
    _, probability_vector = min(
        zip(eigenvalues, eigenvectors.T), key=lambda v: abs(v[0] - 1.0)
    )
    # normalise, this is a probability distribution
    probability_vector /= np.sum(probability_vector)
    return probability_vector


def _compute_entropy_rate(
    right_stochastic_matrix: np.ndarray, stationary_probability_vector: np.ndarray
) -> float:
    """
    Computes the entropy rate in [nat] for a Markov random walk process.

    Arguments:
        right_stochastic_matrix (numpy.ndarray): the stochastic matrix of the Markov
            random walk with each row normalised (summing up to 1)
        stationary_probability_vector: numpy.ndarray): the vector containing stationary
            probability distribution of the Markov process

    Returns:
        float: the entropy rate
    """
    # we ignore the division warning because 0 x log(0) = 0 in information theory
    with np.errstate(divide="ignore"):
        entropy_rate = np.sum(
            stationary_probability_vector
            * np.sum(
                -right_stochastic_matrix
                * np.nan_to_num(np.log(right_stochastic_matrix)),
                axis=1,
            )
        )
    return entropy_rate


def get_entropy_rate(graph: nx.Graph) -> float:
    """
    Compute entropy rate for a given graph.
    https://en.wikipedia.org/wiki/Entropy_rate
    This is under the assumption that we are interested in a Markov random
    walk on the trade graph.

    Arguments:
        graph (nx.Graph): The networkx graph object, presumably a trade graph
            from a PyTradeShifts instance.

    Returns:
        float: the entropy rate of a random walker on the scnearios' trade graph.
    """
    # get the right stochastic matrix and the stationary probabiltiy vector
    stochastic_matrix = get_right_stochastic_matrix(graph)
    probability_vector = get_stationary_probability_vector(stochastic_matrix)
    entropy_rate = _compute_entropy_rate(stochastic_matrix, probability_vector)
    if np.real(entropy_rate) != np.real_if_close(entropy_rate):
        print("Warning: a significant imaginary part encountered in entropy rate:")
        print(entropy_rate)
        print("Returning real part only.")
    return np.real(entropy_rate)


def get_dict_min_max(iterable: dict) -> tuple[Any, Any, Any, Any]:
    """
    Finds minimum and maximum values in a dictionary.

    Arguments:
        iterable (dict): the dictionary in which the search is performed

    Returns:
        tuple[Any, Any, Any, Any]: a tuple containing:
            (key of the smallest value, the smallest value,
            key of the larges value, the largest value)
    """
    max_key = max(iterable, key=iterable.get)
    min_key = min(iterable, key=iterable.get)
    return min_key, iterable[min_key], max_key, iterable[max_key]


def get_graph_efficiency(graph: nx.Graph, normalisation: str | None = "weak") -> float:
    """
    Computes graph efficiency score for the specified graph, based on:
    Bertagnolli, G., Gallotti, R., & De Domenico, M. (2021).
    Quantifying efficient information exchange in real network flows.
    Communications Physics, 4(1), 125.
    https://www.nature.com/articles/s42005-021-00612-5.

    Arguments:
        graph (nx.Graph): The graph for which to compute the efficiency metric.
        Note: we're assuming that all edges have a `1/weight` attribute.
        normalisation (str | None, opional): Whether to normalise (and if so how)
            the efficiency score. If `None` no normalisation occurs, if `strong`
            the score is divided by the efficiency of an ideal flow graph, if
            `weak` the division is by the score for the average of the graph
            and the ideal graph.

    Returns:
        float: The efficiency score.
    """
    all_pairs_paths = dict(nx.all_pairs_dijkstra(graph, weight="1/weight"))
    cost_matrix = pd.DataFrame(
        0.0,
        index=graph.nodes(),
        columns=graph.nodes(),
    )
    flow_matrix = pd.DataFrame(
        0.0,
        index=graph.nodes(),
        columns=graph.nodes(),
    )
    for source, (distances, paths) in all_pairs_paths.items():
        for target, cost in distances.items():
            cost_matrix.loc[source, target] = cost
        for target, path in paths.items():
            flow_matrix.loc[source, target] = nx.path_weight(
                graph, path, weight="weight"
            )
    E = np.sum(1 / cost_matrix.values[cost_matrix != 0])
    match normalisation:
        case "weak":
            ideal_matrix = (flow_matrix + nx.to_pandas_adjacency(graph)) / 2
        case "strong":
            ideal_matrix = flow_matrix
        case None:
            return E
        case _:
            print("Unrecognised normalisation option, defaulting to ``weak''.")
            ideal_matrix = (flow_matrix + nx.to_pandas_adjacency(graph)) / 2
    E_ideal = np.sum(ideal_matrix.values)
    return E / E_ideal


def get_stability_index(
    index_file=f"data{os.sep}stability_index{os.sep}worldbank_governence_indicator_2022_normalised.csv",
) -> dict[str, float]:
    """
    Reads the government stability index from the specified file.
    The format is assumed to be a .csv file with columns: [country, index].

    Arguments:
        index_file (str, optional): Path to the file. By default it leads to
            a World Bank data based index we provide in the repository.

    Returns:
        dict: The mapping of country -> index
    """
    stability_index = pd.read_csv(index_file, index_col=0)
    stability_index.index = coco.convert(stability_index.index, to="name_short")
    return stability_index.loc[:, stability_index.columns[-1]].to_dict()


def get_distance_matrix(index: pd.Index, columns: pd.Index) -> pd.DataFrame | None:
    """
    Computes the distance matrix amongst the centroids of all country pairs
    specified by index and columns of a trade matrix.

    Arguments:
        index (pd.Index): The index of a trade matrix.
        columns (pd.Index): The columns of a trade matrix.

    Returns:
        pd.DataFrame | None: The resulting distance matrix or None if unsuccessful.
    """
    # get central point for each region
    centroids = prepare_centroids(index)
    # compute the distance matrix using a geodesic
    # on an ellipsoidal model of the Earth
    # https://doi.org/10.1007%2Fs00190-012-0578-z
    try:
        distance_matrix = pd.DataFrame(
            squareform(
                pdist(
                    centroids.loc[:, ["latitude", "longitude"]],
                    metric=lambda lat, lon: geodesic(lat, lon).km,
                )
            ),
            columns=columns,
            index=index,
        )
    except ValueError:
        print("Error building the distance matrix.")
        print("Cannot find centroids for these regions:")
        print(index.difference(centroids["name"]))
        return
    return distance_matrix


def get_percolation_eigenvalue(
    adjacency_matrix: np.ndarray, attack_vector: np.ndarray
) -> float:
    """
    Computes the largest eigenvalue of a matrix defined as A_ij*(1-p_i),
    where A is the adjacency matrix of a graph, and p is an attack vector.
    This is described in detail here:
    Restrepo, J. G., Ott, E., & Hunt, B. R. (2008).
    Weighted percolation on directed networks.
    Physical review letters, 100(5), 058701.
    https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.100.058701.

    Arguments:
        adjacency_matrix (np.ndarray): The adjacency matrix A.
        attack_vector (np.ndarray): The attack vector p.

    Returns:
        float: The largest eigenvalue of A_ij*(1-p_i).
    """
    return np.real(
        np.linalg.eigvals((adjacency_matrix.T * (1 - attack_vector)).T).max()
    )


def get_percolation_threshold(
    adjacency_matrix: np.ndarray, node_importance_list: list
) -> tuple[int, list[int], list[float]]:
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

    Arguments:
        adjacency_matrix (np.ndarray): The adjacency matrix A.
        node_importance_list (list): A list of values providing priority to nodes
            to be removed in the order of highest to lowest priority.

    Returns:
        tuple: The threshold value (number of nodes needed to collapse the graph),
            list of consecutive node removals, corresponding eigenvalues list.
    """
    eigenvalues = []
    removed_nodes_count = []
    for node_importance in sorted(node_importance_list, reverse=True):
        attack_vector = np.fromiter(
            map(lambda x: x >= node_importance, node_importance_list),
            dtype=float,
            count=len(adjacency_matrix),
        )
        # largest eigenvalue of a matrix with elements: A_ij * (1-p_i)
        # where A is the adj. matrix, and p is the attack vector
        eigenvalues.append(get_percolation_eigenvalue(adjacency_matrix, attack_vector))
        removed_nodes_count.append(int(attack_vector.sum()))
    # eigenvalue = 1 is the percolation threshold
    threshold = removed_nodes_count[
        len(eigenvalues) - np.searchsorted(eigenvalues[::-1], 1, side="left")
    ]
    return threshold, removed_nodes_count, eigenvalues


def fill_sector_by_colour(
    ax: Axes,
    z_threshold: float,
    p_thresholds: list[float],
    alpha: float,
    labels=True,
    fontsize=7,
):
    """
    This is a helper function to colour the background in the z-score, participation
    space plot.
    There has been several sectors identified in this attribute space in the following
    article:
    Guimerà, R., & Nunes Amaral, L. A. (2005).
    Functional cartography of complex metabolic networks.
    Nature, 433(7028), 895-900. doi:10.1038/nature03288
    https://www.nature.com/articles/nature03288
    Here, we allow for arbitrary thresholds being used since they do depend on
    the dataset and do not necessarily match the values in the aforementoned work.

    Arguments:
        ax (Axes): the axis on which to plot the sectors.
        z_threshold (float): value of z-score for a country to
                be considered a hub.
        p_threshold (list[float]): list of participation
            coefficient threshold for a country to be provincial, connector,
            or kinless (and sub-categories, in the paper cited there are 7 different
            thresholds).
        alpha (float): transparency level of the background colouring,
            showing the sectors defined by the above thresholds.
        labels (bool | True): labels for the aforementioned
            sectors, if true, they will be shown in the plot
        fontsize (int, optional): font size for the labels in the plot.

    Returns:
        None
    """
    x_low, x_high = (0, 1)  # valid range for p
    y_low, y_high = (-2, 6)  # z can be any value technically
    color_cycle = ["purple", "red", "blue", "green", "orange", "gold", "brown"]
    assert len(p_thresholds) >= 2
    # left edge panels
    ii = 0
    ax.fill(
        "x",
        "y",
        color_cycle[ii],
        data={
            "x": [x_low, p_thresholds[ii], p_thresholds[ii], x_low],
            "y": [y_low, y_low, z_threshold, z_threshold],
        },
        alpha=alpha,
    )
    ii += 1
    ax.fill(
        "x",
        "y",
        color_cycle[ii],
        data={
            "x": [x_low, p_thresholds[ii], p_thresholds[ii], x_low],
            "y": [z_threshold, z_threshold, y_high, y_high],
        },
        alpha=alpha,
    )
    # middle panels
    while ii < len(p_thresholds) - 1:
        ii += 1
        ax.fill(
            "x",
            "y",
            color_cycle[ii],
            data={
                "x": [
                    p_thresholds[ii - 2],
                    p_thresholds[ii],
                    p_thresholds[ii],
                    p_thresholds[ii - 2],
                ],
                "y": (
                    [y_low, y_low, z_threshold, z_threshold]
                    if ii % 2 == 0
                    else [z_threshold, z_threshold, y_high, y_high]
                ),
            },
            alpha=alpha,
        )
    # right edge panels
    ax.fill(
        "x",
        "y",
        color_cycle[ii + 1],
        data={
            "x": [
                p_thresholds[ii - 1],
                x_high,
                x_high,
                p_thresholds[ii - 1],
            ],
            "y": [z_threshold, z_threshold, y_high, y_high],
        },
        alpha=alpha,
    )
    ax.fill(
        "x",
        "y",
        color_cycle[ii + 2],
        data={
            "x": [
                p_thresholds[ii],
                x_high,
                x_high,
                p_thresholds[ii],
            ],
            "y": [y_low, y_low, z_threshold, z_threshold],
        },
        alpha=alpha,
    )
    if labels:
        ax.text(0.01, 5.5, "Provincial hub", fontsize=fontsize, color="red", alpha=0.5)
        ax.text(0.31, 5.5, "Connector hub", fontsize=fontsize, color="green", alpha=0.5)
        ax.text(0.76, 5.5, "Kinless hub", fontsize=fontsize, color="gold", alpha=1)
        ax.text(
            0.001,
            -1.8,
            "Ultra\nperi-\npheral\nnon\nhub",
            fontsize=fontsize,
            color="purple",
            alpha=0.5,
        )
        ax.text(
            0.06, -1.8, "Peripheral non-hub", fontsize=fontsize, color="blue", alpha=0.5
        )
        ax.text(
            0.63, -1.8, "Connector non-hub", fontsize=fontsize, color="orange", alpha=1
        )
        ax.text(
            0.81, -1.8, "Kinless non-hub", fontsize=fontsize, color="brown", alpha=1
        )
