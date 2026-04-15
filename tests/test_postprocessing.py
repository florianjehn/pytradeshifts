import pytest
from src.model import PyTradeShifts
from src.postprocessing import Postprocessing
import pandas as pd


class TestPostprocessing:
    """
    Testing class containing tests for Postprocessing.
    Most of actual computation is done (and tested) in utils,
    so here we mostly assert that the data types and shapes.
    """

    @pytest.fixture
    def postprocessing_object(self) -> Postprocessing:
        ISIMIP = pd.read_csv(
            "." + "/data/scenario_files/ISIMIP_climate/ISIMIP_wheat_Hedlung.csv",
            index_col=0,
        )
        nan_indices = ISIMIP.index[ISIMIP.iloc[:, 0].isnull()].tolist()
        Wheat2018 = PyTradeShifts(
            "Wheat",
            2018,
            region="Global",
            testing=False,
            countries_to_remove=nan_indices,
            cd_kwargs={"seed": 2},
            make_plot=False,
        )
        ISIMIP = PyTradeShifts(
            crop="Wheat",
            base_year=2018,
            scenario_file_name="ISIMIP_climate/ISIMIP_wheat_Hedlung.csv",
            scenario_name="ISIMIP",
            countries_to_remove=nan_indices,
            cd_kwargs={"seed": 2},
            make_plot=False,
        )
        return Postprocessing(
            [
                Wheat2018,
                ISIMIP,
            ],
            random_attack_sample_size=2,
        )

    def test_frobenius(self, postprocessing_object) -> None:
        assert postprocessing_object.frobenius is not None
        assert (
            len(postprocessing_object.frobenius)
            == len(postprocessing_object.scenarios) - 1
        )
        assert all([isinstance(f, float) for f in postprocessing_object.frobenius])

    def test_compute_stationary_markov_distance(self, postprocessing_object) -> None:
        assert postprocessing_object.markov is not None
        assert (
            len(postprocessing_object.markov)
            == len(postprocessing_object.scenarios) - 1
        )
        assert all([isinstance(m, float) for m in postprocessing_object.markov])

    def test_compute_entropy_rate_distance(self, postprocessing_object) -> None:
        assert postprocessing_object.entropy_rate is not None
        assert (
            len(postprocessing_object.entropy_rate)
            == len(postprocessing_object.scenarios) - 1
        )
        assert all([isinstance(m, float) for m in postprocessing_object.entropy_rate])

    def test_format_distance_dataframe(self, postprocessing_object) -> None:
        assert postprocessing_object.distance_df is not None
        assert postprocessing_object.distance_df.shape == (
            len(postprocessing_object.scenarios) - 1,
            4,
        )
        assert (~postprocessing_object.distance_df.isna()).all(axis=None)

    def test_compute_centrality(self, postprocessing_object) -> None:
        assert postprocessing_object.in_degree is not None
        assert postprocessing_object.out_degree is not None
        assert isinstance(postprocessing_object.in_degree, list)
        assert all([isinstance(im, dict) for im in postprocessing_object.in_degree])
        assert all(
            [
                isinstance(ik, str) and isinstance(iv, float)
                for im in postprocessing_object.in_degree
                for ik, iv in im.items()
            ]
        )
        assert isinstance(postprocessing_object.out_degree, list)
        assert all([isinstance(om, dict) for om in postprocessing_object.out_degree])
        assert all(
            [
                isinstance(ok, str) and isinstance(ov, float)
                for om in postprocessing_object.out_degree
                for ok, ov in om.items()
            ]
        )

    def test_compute_global_centrality_metrics(self, postprocessing_object) -> None:
        assert postprocessing_object.global_centrality_metrics is not None
        assert len(postprocessing_object.global_centrality_metrics) == len(
            postprocessing_object.scenarios
        )
        assert all(
            [len(m) == 9 for m in postprocessing_object.global_centrality_metrics]
        )

    def test_compute_community_centrality_metrics(self, postprocessing_object) -> None:
        assert postprocessing_object.community_centrality_metrics is not None
        assert len(postprocessing_object.community_centrality_metrics) == len(
            postprocessing_object.scenarios
        )
        assert all(
            [
                len(m) == 9
                for c in postprocessing_object.community_centrality_metrics
                for m in c
            ]
        )

    def test_compute_satisfaction(self, postprocessing_object) -> None:
        assert postprocessing_object.community_satisfaction is not None
        assert postprocessing_object.community_satisfaction_difference is not None
        assert len(postprocessing_object.community_satisfaction) == len(
            postprocessing_object.scenarios
        )
        assert (
            len(postprocessing_object.community_satisfaction_difference)
            == len(postprocessing_object.scenarios) - 1
        )
        assert all(
            [
                isinstance(v, float) or isinstance(v, int)
                for d in postprocessing_object.community_satisfaction
                for v in d.values()
            ],
        )
        assert all(
            [
                isinstance(v, float) or isinstance(v, int)
                for d in postprocessing_object.community_satisfaction_difference
                for v in d.values()
            ],
        )

    def test_network_metrics(self, postprocessing_object) -> None:
        metrics = [
            *[
                (idx, "efficiency", efficiency)
                for idx, efficiency in enumerate(postprocessing_object.efficiency)
            ],
            *[
                (idx, "clustering", clustering)
                for idx, clustering in enumerate(postprocessing_object.clustering)
            ],
            *[
                (idx, "betweenness", betweenness)
                for idx, betweenness in enumerate(postprocessing_object.betweenness)
            ],
            *[
                (idx, "stability", stability)
                for idx, stability in enumerate(postprocessing_object.network_stability)
            ],
        ]
        metrics.extend(
            [
                (
                    idx,
                    attack,
                    threshold,
                )
                for idx, scenario in enumerate(postprocessing_object.percolation)
                for attack, (threshold, _, _) in scenario.items()
            ]
        )
        assert all([isinstance(idx, int) for (idx, _, _) in metrics])
        assert all([isinstance(metric_name, str) for (_, metric_name, _) in metrics])
        assert all(
            [
                isinstance(metric_value, float) or isinstance(metric_value, int)
                for (_, _, metric_value) in metrics
            ]
        )

    def test_compute_within_community_degree(self, postprocessing_object) -> None:
        assert postprocessing_object.zscores is not None
        assert len(postprocessing_object.zscores) == len(
            postprocessing_object.scenarios
        )
        assert all(
            [
                isinstance(v, float)
                for sc in postprocessing_object.zscores
                for v in sc.values()
            ]
        )

    def test_compute_participation(self, postprocessing_object) -> None:
        assert postprocessing_object.participation is not None
        assert len(postprocessing_object.participation) == len(
            postprocessing_object.scenarios
        )
        assert all(
            [
                isinstance(v, float)
                for sc in postprocessing_object.participation
                for v in sc.values()
            ]
        )
        assert all(
            [
                v >= 0.0 and v <= 1.0
                for sc in postprocessing_object.participation
                for v in sc.values()
            ]
        )

    def test_compute_imports(self, postprocessing_object) -> None:
        assert postprocessing_object.imports is not None
        assert len(postprocessing_object.imports) == len(
            postprocessing_object.scenarios
        )
        assert all(
            [
                isinstance(v, float)
                for sc in postprocessing_object.imports
                for v in sc.values()
            ]
        )

    def test_compute_import_change_as_fraction_of_consumption(
        self, postprocessing_object
    ) -> None:
        # Build a consumption Series from the base scenario's trade matrix
        # (use column sums as a stand-in for "consumption" so every country has data)
        base_trade = postprocessing_object.scenarios[0].trade_matrix
        consumption = base_trade.sum(axis=0)
        # Replace zeros to avoid NaN results
        consumption = consumption.replace(0, 1)

        postprocessing_object.consumption_data = consumption
        postprocessing_object._compute_import_change_as_fraction_of_consumption()

        result = postprocessing_object.import_change_as_fraction_of_consumption
        assert result is not None
        assert len(result) == len(postprocessing_object.scenarios) - 1
        assert all(isinstance(result[0][c], float) for c in result[0])


def test_find_new_order() -> None:
    """
    Test the community rearranging function that is supposed to make sure
    the anchor countries' communities are in order of the passed "anchor_coutries"
    parameter.
    """
    Wheat2018 = PyTradeShifts(
        "Wheat",
        2018,
        region="Global",
        make_plot=False,
    )
    pp = Postprocessing([Wheat2018], testing=True, anchor_countries=["China", "Russia"])
    new_order = pp._find_new_order(pp.scenarios[0])
    assert "China" in new_order[0]
    assert "Russia" in new_order[1]


if __name__ == "__main__":
    ISIMIP = pd.read_csv(
        "." + "/data/scenario_files/ISIMIP_climate/ISIMIP_wheat_Hedlung.csv",
        index_col=0,
    )
    nan_indices = ISIMIP.index[ISIMIP.iloc[:, 0].isnull()].tolist()
    Wheat2018 = PyTradeShifts(
        "Wheat",
        2018,
        region="Global",
        testing=False,
        countries_to_remove=nan_indices,
        cd_kwargs={"seed": 2},
        make_plot=False,
    )
    ISIMIP = PyTradeShifts(
        crop="Wheat",
        base_year=2018,
        scenario_file_name="ISIMIP_climate/ISIMIP_wheat_Hedlung.csv",
        scenario_name="ISIMIP",
        countries_to_remove=nan_indices,
        cd_kwargs={"seed": 2},
        make_plot=False,
    )
    pp = Postprocessing(
        [
            Wheat2018,
            ISIMIP,
        ],
        random_attack_sample_size=20,
    )
    pp.report(utc=True)
