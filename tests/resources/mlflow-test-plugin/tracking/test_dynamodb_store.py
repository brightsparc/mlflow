#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import random
import six
import string
import time
import unittest
import uuid

import mock
import pytest

from moto.dynamodb2 import mock_dynamodb2

from mlflow.entities.lifecycle_stage import LifecycleStage
from mlflow.entities import (
    Metric,
    Param,
    RunTag,
    ViewType,
    LifecycleStage,
    RunStatus,
    RunData,
    ExperimentTag,
)
from mlflow.exceptions import MlflowException
from mlflow.utils.env import get_env
from mlflow.store.tracking import SEARCH_MAX_RESULTS_DEFAULT

from tracking.dynamodb_store import DynamodbStore

_DYNAMODB_ENDPOINT_URL_VAR = "MLFLOW_DYNAMODB_ENDPOINT_URL"


def _default_endpoint_url():
    return get_env(_DYNAMODB_ENDPOINT_URL_VAR)


def random_int(lo=1, hi=1e10):
    return random.randint(lo, hi)


def random_str(size=10, chars=string.ascii_uppercase + string.digits):
    return "".join(random.choice(chars) for _ in range(size))


class TestDynamodbStore(unittest.TestCase):
    DEFAULT_EXPERIMENT_ID = "0"

    def setUp(self):
        # Test with localhost to test global secondary indices / projections
        self.endpoint_url = _default_endpoint_url()
        if self.endpoint_url:
            print("using local dynamodb: {}".format(self.endpoint_url))
            self.region_name = None
            self.use_gsi = True
            self.use_projections = True
        else:
            print("using mock dynamodb")
            self.region_name = "us-west-1"
            self.use_gsi = False
            self.use_projections = False
            # Create a mock dynamodb table in bucket in moto
            # Note that we must set these as environment variables in case users
            # so that boto does not attempt to assume credentials from the ~/.aws/config
            # or IAM role. moto does not correctly pass the arguments to boto3.client().
            os.environ["AWS_ACCESS_KEY_ID"] = "a"
            os.environ["AWS_SECRET_ACCESS_KEY"] = "b"
            self.mock = mock_dynamodb2()
            self.mock.start()

        self.table_prefix = "mlflow"
        self.store = DynamodbStore(
            store_uri="dynamodb:{}".format(self.table_prefix),
            endpoint_url=self.endpoint_url,
            region_name=self.region_name,
            use_gsi=self.use_gsi,
            use_projections=self.use_projections,
            create_tables=True,
        )
        self._populate_tables()

    def tearDown(self):
        if self.endpoint_url:
            self.store.delete_tables()
        else:
            self.mock.stop()

    def _get_store(self):
        return self.store

    def _write_table(self, name, d):
        dynamodb = self._get_store()._get_dynamodb_resource()
        table_name = "_".join([self.table_prefix, name])
        table = dynamodb.Table(table_name)
        table.put_item(Item=d)

    def _populate_tables(
        self, exp_count=3, run_count=2, param_count=5, metric_count=3, values_count=10
    ):
        print("populate tables")
        self.experiments = [str(random_int(100, int(1e9))) for _ in range(exp_count)]
        self.exp_data = {}
        self.run_data = {}
        self.experiments.append(TestDynamodbStore.DEFAULT_EXPERIMENT_ID)
        for exp in self.experiments:
            # create experiment
            exp_folder = os.path.join(self.table_prefix, exp)
            d = {
                "experiment_id": exp,
                "name": random_str(),
                "artifact_location": exp_folder,
                "lifecycle_stage": LifecycleStage.ACTIVE,  # Required for tests
            }
            self.exp_data[exp] = d
            self._write_table(DynamodbStore.EXPERIMENT_TABLE, d)
            # add runs
            self.exp_data[exp]["runs"] = []
            for _ in range(run_count):
                run_id = uuid.uuid4().hex
                self.exp_data[exp]["runs"].append(run_id)
                run_folder = os.path.join(exp_folder, run_id)
                run_info = {
                    "run_uuid": run_id,
                    "run_id": run_id,
                    "experiment_id": exp,
                    "user_id": random_str(random_int(10, 25)),
                    "status": random.choice(RunStatus.all_status()),
                    "start_time": random_int(1, 10),
                    "end_time": random_int(20, 30),
                    "tags": [],
                    "artifact_uri": "{}/artifacts".format(run_folder),
                    "lifecycle_stage": LifecycleStage.ACTIVE,  # Required for tests
                }
                self._write_table("run", run_info)
                self.run_data[run_id] = run_info
                # params
                params = {}
                for _ in range(param_count):
                    param_name = random_str(random_int(4, 12))
                    param_value = random_str(random_int(10, 15))
                    self._write_table(
                        "run_param", {"run_id": run_id, "key": param_name, "value": param_value},
                    )
                    params[param_name] = param_value
                self.run_data[run_id]["params"] = params
                # metrics
                metrics = {}
                for _ in range(metric_count):
                    metric_name = random_str(random_int(6, 10))
                    timestamp = int(time.time())
                    values, values_map = [], []
                    for i in range(values_count):
                        metric_value = random_int(i * 100, (i * 1) * 100)
                        timestamp += random_int(i * 1000, (i + 1) * 1000)
                        values.append((timestamp, metric_value))
                        values_map.insert(0, {"timestamp": timestamp, "value": metric_value})
                    self._write_table(
                        "run_metric", {"run_id": run_id, "key": metric_name, "metrics": values_map},
                    )
                    metrics[metric_name] = values
                self.run_data[run_id]["metrics"] = metrics

    def test_list_experiments(self):
        fs = self._get_store()
        for exp in fs.list_experiments():
            exp_id = exp.experiment_id
            self.assertTrue(exp_id in self.experiments)
            self.assertEqual(exp.name, self.exp_data[exp_id]["name"])
            self.assertEqual(exp.artifact_location, self.exp_data[exp_id]["artifact_location"])

    def _verify_experiment(self, fs, exp_id):
        exp = fs.get_experiment(exp_id)
        self.assertEqual(exp.experiment_id, exp_id)
        self.assertEqual(exp.name, self.exp_data[exp_id]["name"])
        self.assertEqual(exp.artifact_location, self.exp_data[exp_id]["artifact_location"])

    def test_get_experiment(self):
        fs = self._get_store()
        for exp_id in self.experiments:
            self._verify_experiment(fs, exp_id)

        # test that fake experiments dont exist.
        # look for random experiment ids between 8000, 15000 since created ones are (100, 2000)
        for exp_id in set(random_int(8000, 15000) for x in range(20)):
            with self.assertRaises(Exception):
                fs.get_experiment(exp_id)

    def test_get_experiment_by_name(self):
        fs = self._get_store()
        for exp_id in self.experiments:
            name = self.exp_data[exp_id]["name"]
            exp = fs.get_experiment_by_name(name)
            self.assertEqual(exp.experiment_id, exp_id)
            self.assertEqual(exp.name, self.exp_data[exp_id]["name"])
            self.assertEqual(exp.artifact_location, self.exp_data[exp_id]["artifact_location"])

        # test that fake experiments dont exist.
        # look up experiments with names of length 15 since created ones are of length 10
        for exp_names in set(random_str(15) for x in range(20)):
            exp = fs.get_experiment_by_name(exp_names)
            self.assertIsNone(exp)

    def test_create_first_experiment(self):
        fs = self._get_store()
        fs.list_experiments = mock.Mock(return_value=[])
        fs._create_experiment_with_id = mock.Mock()
        fs.create_experiment(random_str(1))
        fs._create_experiment_with_id.assert_called_once()
        experiment_id = fs._create_experiment_with_id.call_args[0][1]
        self.assertEqual(experiment_id, TestDynamodbStore.DEFAULT_EXPERIMENT_ID)

    def test_create_experiment(self):
        fs = self._get_store()

        # Error cases
        with self.assertRaises(Exception):
            fs.create_experiment(None)
        with self.assertRaises(Exception):
            fs.create_experiment("")

        name = random_str(25)  # since existing experiments are 10 chars long
        created_id = fs.create_experiment(name)
        # get the new experiment (by id) and verify (by name)
        exp1 = fs.get_experiment(created_id)
        self.assertEqual(exp1.name, name)

        # get the new experiment (by name) and verify (by id)
        exp2 = fs.get_experiment_by_name(name)
        self.assertEqual(exp2.experiment_id, created_id)

    def test_create_duplicate_experiments(self):
        fs = self._get_store()
        for exp_id in self.experiments:
            name = self.exp_data[exp_id]["name"]
            with self.assertRaises(Exception):
                fs.create_experiment(name)

    def _extract_ids(self, experiments):
        return [e.experiment_id for e in experiments]

    def test_delete_restore_experiment(self):
        fs = self._get_store()
        exp_id = self.experiments[random_int(0, len(self.experiments) - 1)]
        exp_name = self.exp_data[exp_id]["name"]

        # delete it
        fs.delete_experiment(exp_id)
        self.assertTrue(exp_id not in self._extract_ids(fs.list_experiments(ViewType.ACTIVE_ONLY)))
        self.assertTrue(exp_id in self._extract_ids(fs.list_experiments(ViewType.DELETED_ONLY)))
        self.assertTrue(exp_id in self._extract_ids(fs.list_experiments(ViewType.ALL)))
        self.assertEqual(fs.get_experiment(exp_id).lifecycle_stage, LifecycleStage.DELETED)

        # restore it
        fs.restore_experiment(exp_id)
        restored_1 = fs.get_experiment(exp_id)
        self.assertEqual(restored_1.experiment_id, exp_id)
        self.assertEqual(restored_1.name, exp_name)
        restored_2 = fs.get_experiment_by_name(exp_name)
        self.assertEqual(restored_2.experiment_id, exp_id)
        self.assertEqual(restored_2.name, exp_name)
        self.assertTrue(exp_id in self._extract_ids(fs.list_experiments(ViewType.ACTIVE_ONLY)))
        self.assertTrue(exp_id not in self._extract_ids(fs.list_experiments(ViewType.DELETED_ONLY)))
        self.assertTrue(exp_id in self._extract_ids(fs.list_experiments(ViewType.ALL)))
        self.assertEqual(fs.get_experiment(exp_id).lifecycle_stage, LifecycleStage.ACTIVE)

    def test_rename_experiment(self):
        fs = self._get_store()
        exp_id = self.experiments[random_int(0, len(self.experiments) - 1)]
        exp_name = self.exp_data[exp_id]["name"]
        new_name = exp_name + "!!!"
        self.assertNotEqual(exp_name, new_name)
        self.assertEqual(fs.get_experiment(exp_id).name, exp_name)
        fs.rename_experiment(exp_id, new_name)
        self.assertEqual(fs.get_experiment(exp_id).name, new_name)

        # Ensure that we cannot rename deleted experiments.
        fs.delete_experiment(exp_id)
        with pytest.raises(Exception) as e:
            fs.rename_experiment(exp_id, exp_name)
        assert "non-active lifecycle" in str(e.value)
        self.assertEqual(fs.get_experiment(exp_id).name, new_name)

        # Restore the experiment, and confirm that we acn now rename it.
        fs.restore_experiment(exp_id)
        self.assertEqual(fs.get_experiment(exp_id).name, new_name)
        fs.rename_experiment(exp_id, exp_name)
        self.assertEqual(fs.get_experiment(exp_id).name, exp_name)

    def test_delete_restore_run(self):
        fs = self._get_store()
        exp_id = self.experiments[random_int(0, len(self.experiments) - 1)]
        run_id = self.exp_data[exp_id]["runs"][0]
        # Should not throw.
        assert fs.get_run(run_id).info.lifecycle_stage == "active"
        fs.delete_run(run_id)
        assert fs.get_run(run_id).info.lifecycle_stage == "deleted"
        fs.restore_run(run_id)
        assert fs.get_run(run_id).info.lifecycle_stage == "active"

    def test_create_run_in_deleted_experiment(self):
        fs = self._get_store()
        exp_id = self.experiments[random_int(0, len(self.experiments) - 1)]
        # delete it
        fs.delete_experiment(exp_id)
        with pytest.raises(Exception):
            fs.create_run(exp_id, "user", 0, [])

    def test_create_run_returns_expected_run_data(self):
        fs = self._get_store()
        no_tags_run = fs.create_run(
            experiment_id=TestDynamodbStore.DEFAULT_EXPERIMENT_ID,
            user_id="user",
            start_time=0,
            tags=[],
        )
        assert isinstance(no_tags_run.data, RunData)
        assert len(no_tags_run.data.tags) == 0

        tags_dict = {
            "my_first_tag": "first",
            "my-second-tag": "2nd",
        }
        tags_entities = [RunTag(key, value) for key, value in tags_dict.items()]
        tags_run = fs.create_run(
            experiment_id=TestDynamodbStore.DEFAULT_EXPERIMENT_ID,
            user_id="user",
            start_time=0,
            tags=tags_entities,
        )
        assert isinstance(tags_run.data, RunData)
        assert tags_run.data.tags == tags_dict

    def _experiment_id_edit_func(self, old_dict):
        old_dict["experiment_id"] = int(old_dict["experiment_id"])
        return old_dict

    def _verify_run(self, fs, run_id):
        run = fs.get_run(run_id)
        run_info = self.run_data[run_id]
        run_info.pop("metrics", None)
        run_info.pop("params", None)
        run_info.pop("tags", None)
        run_info["lifecycle_stage"] = LifecycleStage.ACTIVE
        run_info["status"] = RunStatus.to_string(run_info["status"])
        self.assertEqual(run_info, dict(run.info))

    def test_get_run(self):
        fs = self._get_store()
        for exp_id in self.experiments:
            runs = self.exp_data[exp_id]["runs"]
            for run_id in runs:
                self._verify_run(fs, run_id)

    def test_list_run_infos(self):
        fs = self._get_store()
        for exp_id in self.experiments:
            run_infos = fs.list_run_infos(exp_id, run_view_type=ViewType.ALL)
            for run_info in run_infos:
                run_id = run_info.run_id
                dict_run_info = self.run_data[run_id]
                dict_run_info.pop("metrics")
                dict_run_info.pop("params")
                dict_run_info.pop("tags")
                dict_run_info["lifecycle_stage"] = LifecycleStage.ACTIVE
                dict_run_info["status"] = RunStatus.to_string(dict_run_info["status"])
                self.assertEqual(dict_run_info, dict(run_info))

    @pytest.mark.skip(reason="TODO: Fix getting latest metric when not retrieving history")
    def test_log_metric_allows_multiple_values_at_same_step_and_run_data_uses_max_step_value(self):
        fs = self._get_store()
        run_id = self._create_run(fs).info.run_id

        metric_name = "test-metric-1"
        # Check that we get the max of (step, timestamp, value) in that order
        tuples_to_log = [
            (0, 100, 1000),
            (3, 40, 100),  # larger step wins even though it has smaller value
            (3, 50, 10),  # larger timestamp wins even though it has smaller value
            (3, 50, 20),  # tiebreak by max value
            (3, 50, 20),  # duplicate metrics with same (step, timestamp, value) are ok
            # verify that we can log steps out of order / negative steps
            (-3, 900, 900),
            (-1, 800, 800),
        ]
        for step, timestamp, value in reversed(tuples_to_log):
            fs.log_metric(run_id, Metric(metric_name, value, timestamp, step))

        metric_history = fs.get_metric_history(run_id, metric_name)
        logged_tuples = [(m.step, m.timestamp, m.value) for m in metric_history]
        assert set(logged_tuples) == set(tuples_to_log)

        run_data = fs.get_run(run_id).data
        print("run data", run_data)
        run_metrics = run_data.metrics
        print("run metrics", run_metrics)

        assert len(run_metrics) == 1
        assert run_metrics[metric_name] == 20
        metric_obj = run_data._metric_objs[0]
        assert metric_obj.key == metric_name
        assert metric_obj.step == 3
        assert metric_obj.timestamp == 50
        assert metric_obj.value == 20

    @pytest.mark.skip(reason="Investigate why doesn't work with moto")
    def test_get_all_metrics(self):
        fs = self._get_store()
        for exp_id in self.experiments:
            runs = self.exp_data[exp_id]["runs"]
            for run_id in runs:
                run_info = self.run_data[run_id]
                metrics = fs.get_all_metrics(run_id)
                metrics_dict = run_info.pop("metrics")
                for metric in metrics:
                    expected_timestamp, expected_value = max(metrics_dict[metric.key])
                    self.assertEqual(metric.timestamp, expected_timestamp)
                    self.assertEqual(metric.value, expected_value)

    def skip(self):
        fs = self._get_store()
        for exp_id in self.experiments:
            runs = self.exp_data[exp_id]["runs"]
            for run_id in runs:
                run_info = self.run_data[run_id]
                metrics = run_info.pop("metrics")
                for metric_name, values in metrics.items():
                    metric_history = fs.get_metric_history(run_id, metric_name)
                    sorted_values = sorted(values, reverse=True)
                    for metric in metric_history:
                        timestamp, metric_value = sorted_values.pop()
                        self.assertEqual(metric.timestamp, timestamp)
                        self.assertEqual(metric.key, metric_name)
                        self.assertEqual(metric.value, metric_value)

    def _search(
        self,
        fs,
        experiment_id,
        filter_str=None,
        run_view_type=ViewType.ALL,
        max_results=SEARCH_MAX_RESULTS_DEFAULT,
    ):
        return [
            r.info.run_id
            for r in fs.search_runs([experiment_id], filter_str, run_view_type, max_results)
        ]

    def test_search_runs(self):
        # replace with test with code is implemented
        fs = self._get_store()
        # Expect 2 runs for each experiment
        assert len(self._search(fs, self.experiments[0], run_view_type=ViewType.ACTIVE_ONLY)) == 2
        assert len(self._search(fs, self.experiments[0])) == 2
        assert len(self._search(fs, self.experiments[0], run_view_type=ViewType.DELETED_ONLY)) == 0

    def test_search_tags(self):
        fs = self._get_store()
        experiment_id = self.experiments[0]
        r1 = fs.create_run(experiment_id, "user", 0, []).info.run_id
        r2 = fs.create_run(experiment_id, "user", 0, []).info.run_id

        fs.set_tag(r1, RunTag("generic_tag", "p_val"))
        fs.set_tag(r2, RunTag("generic_tag", "p_val"))

        fs.set_tag(r1, RunTag("generic_2", "some value"))
        fs.set_tag(r2, RunTag("generic_2", "another value"))

        fs.set_tag(r1, RunTag("p_a", "abc"))
        fs.set_tag(r2, RunTag("p_b", "ABC"))

        # test search returns both runs
        six.assertCountEqual(
            self, [r1, r2], self._search(fs, experiment_id, filter_str="tags.generic_tag = 'p_val'")
        )
        # test search returns appropriate run (same key different values per run)
        six.assertCountEqual(
            self, [r1], self._search(fs, experiment_id, filter_str="tags.generic_2 = 'some value'")
        )
        six.assertCountEqual(
            self, [r2], self._search(fs, experiment_id, filter_str="tags.generic_2='another value'")
        )
        six.assertCountEqual(
            self, [], self._search(fs, experiment_id, filter_str="tags.generic_tag = 'wrong_val'")
        )
        six.assertCountEqual(
            self, [], self._search(fs, experiment_id, filter_str="tags.generic_tag != 'p_val'")
        )
        six.assertCountEqual(
            self,
            [r1, r2],
            self._search(fs, experiment_id, filter_str="tags.generic_tag != 'wrong_val'"),
        )
        six.assertCountEqual(
            self,
            [r1, r2],
            self._search(fs, experiment_id, filter_str="tags.generic_2 != 'wrong_val'"),
        )
        six.assertCountEqual(
            self, [r1], self._search(fs, experiment_id, filter_str="tags.p_a = 'abc'")
        )
        six.assertCountEqual(
            self, [r2], self._search(fs, experiment_id, filter_str="tags.p_b = 'ABC'")
        )

    def test_search_with_max_results(self):
        fs = self._get_store()
        exp = fs.create_experiment("search_with_max_results")

        runs = [fs.create_run(exp, "user", r, []).info.run_id for r in range(10)]
        runs.reverse()

        print(runs)
        print(self._search(fs, exp))
        assert runs[:10] == self._search(fs, exp)
        for n in [0, 1, 2, 4, 8, 10, 20, 50, 100, 500, 1000, 1200, 2000]:
            assert runs[: min(1200, n)] == self._search(fs, exp, max_results=n)

        with self.assertRaises(MlflowException) as e:
            self._search(fs, exp, None, max_results=int(1e10))
        self.assertIn("Invalid value for request parameter max_results. It ", e.exception.message)

    def test_search_with_deterministic_max_results(self):
        fs = self._get_store()
        exp = fs.create_experiment("test_search_with_deterministic_max_results")

        # Create 10 runs with the same start_time.
        # Sort based on run_id
        runs = sorted([fs.create_run(exp, "user", 1000, []).info.run_id for r in range(10)])
        for n in [0, 1, 2, 4, 8, 10, 20]:
            assert runs[: min(10, n)] == self._search(fs, exp, max_results=n)

    def test_search_runs_pagination(self):
        fs = self._get_store()
        exp = fs.create_experiment("test_search_runs_pagination")
        # test returned token behavior
        runs = sorted([fs.create_run(exp, "user", 1000, []).info.run_id for r in range(10)])
        result = fs.search_runs([exp], None, ViewType.ALL, max_results=4)
        assert [r.info.run_id for r in result] == runs[0:4]
        assert result.token is not None
        result = fs.search_runs([exp], None, ViewType.ALL, max_results=4, page_token=result.token)
        assert [r.info.run_id for r in result] == runs[4:8]
        assert result.token is not None
        result = fs.search_runs([exp], None, ViewType.ALL, max_results=4, page_token=result.token)
        assert [r.info.run_id for r in result] == runs[8:]
        assert result.token is None

    def test_weird_param_names(self):
        WEIRD_PARAM_NAME = "this is/a weird/but valid param"
        fs = self._get_store()
        run_id = self.exp_data[TestDynamodbStore.DEFAULT_EXPERIMENT_ID]["runs"][0]
        fs.log_param(run_id, Param(WEIRD_PARAM_NAME, "Value"))
        run = fs.get_run(run_id)
        assert run.data.params[WEIRD_PARAM_NAME] == "Value"

    @pytest.mark.skip(reason="Empty strings in DynamoDb not supported")
    def test_log_empty_str(self):
        PARAM_NAME = "new param"
        fs = self._get_store()
        run_id = self.exp_data[TestDynamodbStore.DEFAULT_EXPERIMENT_ID]["runs"][0]
        fs.log_param(run_id, Param(PARAM_NAME, ""))
        run = fs.get_run(run_id)
        assert run.data.params[PARAM_NAME] == ""

    def test_weird_metric_names(self):
        WEIRD_METRIC_NAME = "this is/a weird/but valid metric"
        fs = self._get_store()
        run_id = self.exp_data[TestDynamodbStore.DEFAULT_EXPERIMENT_ID]["runs"][0]
        fs.log_metric(run_id, Metric(WEIRD_METRIC_NAME, 10, 1234, 0))
        run = fs.get_run(run_id)
        assert run.data.metrics[WEIRD_METRIC_NAME] == 10
        history = fs.get_metric_history(run_id, WEIRD_METRIC_NAME)
        assert len(history) == 1
        metric = history[0]
        assert metric.key == WEIRD_METRIC_NAME
        assert metric.value == 10
        assert metric.timestamp == 1234

    def test_weird_tag_names(self):
        WEIRD_TAG_NAME = "this is/a weird/but valid tag"
        fs = self._get_store()
        run_id = self.exp_data[TestDynamodbStore.DEFAULT_EXPERIMENT_ID]["runs"][0]
        fs.set_tag(run_id, RunTag(WEIRD_TAG_NAME, "Muhahaha!"))
        run = fs.get_run(run_id)
        assert run.data.tags[WEIRD_TAG_NAME] == "Muhahaha!"

    def test_set_experiment_tags(self):
        fs = self._get_store()
        fs.set_experiment_tag(
            TestDynamodbStore.DEFAULT_EXPERIMENT_ID, ExperimentTag("tag0", "value0")
        )
        fs.set_experiment_tag(
            TestDynamodbStore.DEFAULT_EXPERIMENT_ID, ExperimentTag("tag1", "value1")
        )
        experiment = fs.get_experiment(TestDynamodbStore.DEFAULT_EXPERIMENT_ID)
        assert len(experiment.tags) == 2
        assert experiment.tags["tag0"] == "value0"
        assert experiment.tags["tag1"] == "value1"
        # test that updating a tag works
        fs.set_experiment_tag(
            TestDynamodbStore.DEFAULT_EXPERIMENT_ID, ExperimentTag("tag0", "value00000")
        )
        experiment = fs.get_experiment(TestDynamodbStore.DEFAULT_EXPERIMENT_ID)
        assert experiment.tags["tag0"] == "value00000"
        assert experiment.tags["tag1"] == "value1"
        # test that setting a tag on 1 experiment does not impact another experiment.
        exp_id = None
        for exp in self.experiments:
            if exp != TestDynamodbStore.DEFAULT_EXPERIMENT_ID:
                exp_id = exp
                break
        experiment = fs.get_experiment(exp_id)
        assert len(experiment.tags) == 0
        # setting a tag on different experiments maintains different values across experiments
        fs.set_experiment_tag(exp_id, ExperimentTag("tag1", "value11111"))
        experiment = fs.get_experiment(exp_id)
        assert len(experiment.tags) == 1
        assert experiment.tags["tag1"] == "value11111"
        experiment = fs.get_experiment(TestDynamodbStore.DEFAULT_EXPERIMENT_ID)
        assert experiment.tags["tag0"] == "value00000"
        assert experiment.tags["tag1"] == "value1"
        # test can set multi-line tags
        fs.set_experiment_tag(exp_id, ExperimentTag("multiline_tag", "value2\nvalue2\nvalue2"))
        experiment = fs.get_experiment(exp_id)
        assert experiment.tags["multiline_tag"] == "value2\nvalue2\nvalue2"
        # test cannot set tags on deleted experiments
        fs.delete_experiment(exp_id)
        with pytest.raises(MlflowException):
            fs.set_experiment_tag(exp_id, ExperimentTag("should", "notset"))

    def test_set_tags(self):
        fs = self._get_store()
        run_id = self.exp_data[TestDynamodbStore.DEFAULT_EXPERIMENT_ID]["runs"][0]
        fs.set_tag(run_id, RunTag("tag0", "value0"))
        fs.set_tag(run_id, RunTag("tag1", "value1"))
        tags = fs.get_run(run_id).data.tags
        assert tags["tag0"] == "value0"
        assert tags["tag1"] == "value1"

        # Can overwrite tags.
        fs.set_tag(run_id, RunTag("tag0", "value2"))
        tags = fs.get_run(run_id).data.tags
        assert tags["tag0"] == "value2"
        assert tags["tag1"] == "value1"

        # Can set multiline tags.
        fs.set_tag(run_id, RunTag("multiline_tag", "value2\nvalue2\nvalue2"))
        tags = fs.get_run(run_id).data.tags
        assert tags["multiline_tag"] == "value2\nvalue2\nvalue2"

    def test_delete_tags(self):
        fs = self._get_store()
        exp_id = self.experiments[random_int(0, len(self.experiments) - 1)]
        run_id = self.exp_data[exp_id]["runs"][0]
        fs.set_tag(run_id, RunTag("tag0", "value0"))
        fs.set_tag(run_id, RunTag("tag1", "value1"))
        tags = fs.get_run(run_id).data.tags
        assert tags["tag0"] == "value0"
        assert tags["tag1"] == "value1"
        fs.delete_tag(run_id, "tag0")
        new_tags = fs.get_run(run_id).data.tags
        assert "tag0" not in new_tags.keys()
        # test that you cannot delete tags that don't exist.
        with pytest.raises(MlflowException):
            fs.delete_tag(run_id, "fakeTag")
        # test that you cannot delete tags for nonexistent runs
        with pytest.raises(MlflowException):
            fs.delete_tag("random_id", "tag0")
        fs = self._get_store()
        fs.delete_run(run_id)
        # test that you cannot delete tags for deleted runs.
        assert fs.get_run(run_id).info.lifecycle_stage == LifecycleStage.DELETED
        with pytest.raises(MlflowException):
            fs.delete_tag(run_id, "tag0")

    def test_unicode_tag(self):
        fs = self._get_store()
        run_id = self.exp_data[TestDynamodbStore.DEFAULT_EXPERIMENT_ID]["runs"][0]
        value = u"𝐼 𝓈𝑜𝓁𝑒𝓂𝓃𝓁𝓎 𝓈𝓌𝑒𝒶𝓇 𝓉𝒽𝒶𝓉 𝐼 𝒶𝓂 𝓊𝓅 𝓉𝑜 𝓃𝑜 𝑔𝑜𝑜𝒹"
        fs.set_tag(run_id, RunTag("message", value))
        tags = fs.get_run(run_id).data.tags
        assert tags["message"] == value

    def test_get_deleted_run(self):
        """
        Getting metrics/tags/params/run info should be allowed on deleted runs.
        """
        fs = self._get_store()
        exp_id = self.experiments[random_int(0, len(self.experiments) - 1)]
        run_id = self.exp_data[exp_id]["runs"][0]
        fs.delete_run(run_id)
        assert fs.get_run(run_id)

    def test_set_deleted_run(self):
        """
        Setting metrics/tags/params/updating run info should not be allowed on deleted runs.
        """
        fs = self._get_store()
        exp_id = self.experiments[random_int(0, len(self.experiments) - 1)]
        run_id = self.exp_data[exp_id]["runs"][0]
        fs.delete_run(run_id)

        assert fs.get_run(run_id).info.lifecycle_stage == LifecycleStage.DELETED
        with pytest.raises(MlflowException):
            fs.set_tag(run_id, RunTag("a", "b"))
        with pytest.raises(MlflowException):
            fs.log_metric(run_id, Metric("a", 0.0, timestamp=0, step=0))
        with pytest.raises(MlflowException):
            fs.log_param(run_id, Param("a", "b"))

    def test_default_experiment_initialization(self):
        fs = self._get_store()
        fs.delete_experiment(TestDynamodbStore.DEFAULT_EXPERIMENT_ID)
        fs = self._get_store()
        experiment = fs.get_experiment(TestDynamodbStore.DEFAULT_EXPERIMENT_ID)
        assert experiment.lifecycle_stage == LifecycleStage.DELETED

    @pytest.mark.skip(reason="Experiment cleanup not consistent")
    def test_bad_experiment_id_recorded_for_run(self):
        fs = self._get_store()
        exp_0 = fs.get_experiment(TestDynamodbStore.DEFAULT_EXPERIMENT_ID)
        all_runs = self._search(fs, exp_0.experiment_id)

        all_run_ids = self.exp_data[exp_0.experiment_id]["runs"]
        assert len(all_runs) == len(all_run_ids)

    def test_log_batch(self):
        fs = self._get_store()
        run = fs.create_run(
            experiment_id=TestDynamodbStore.DEFAULT_EXPERIMENT_ID,
            user_id="user",
            start_time=0,
            tags=[],
        )
        run_id = run.info.run_id
        metric_entities = [Metric("m1", 0.87, 12345, 0), Metric("m2", 0.49, 12345, 0)]
        param_entities = [Param("p1", "p1val"), Param("p2", "p2val")]
        tag_entities = [RunTag("t1", "t1val"), RunTag("t2", "t2val")]
        fs.log_batch(
            run_id=run_id, metrics=metric_entities, params=param_entities, tags=tag_entities
        )
        self._verify_logged(fs, run_id, metric_entities, param_entities, tag_entities)

    def _create_run(self, fs):
        return fs.create_run(
            experiment_id=TestDynamodbStore.DEFAULT_EXPERIMENT_ID,
            user_id="user",
            start_time=0,
            tags=[],
        )

    def _verify_logged(self, fs, run_id, metrics, params, tags):
        run = fs.get_run(run_id)
        all_metrics = sum([fs.get_metric_history(run_id, key) for key in run.data.metrics], [])
        assert len(all_metrics) == len(metrics)
        logged_metrics = [(m.key, m.value, m.timestamp, m.step) for m in all_metrics]
        assert set(logged_metrics) == set([(m.key, m.value, m.timestamp, m.step) for m in metrics])
        logged_tags = set([(tag_key, tag_value) for tag_key, tag_value in run.data.tags.items()])
        assert set([(tag.key, tag.value) for tag in tags]) <= logged_tags
        assert len(run.data.params) == len(params)
        logged_params = [(param_key, param_val) for param_key, param_val in run.data.params.items()]
        assert set(logged_params) == set([(param.key, param.value) for param in params])

    def test_log_batch_nonexistent_run(self):
        fs = self._get_store()
        nonexistent_uuid = uuid.uuid4().hex
        with self.assertRaises(MlflowException) as e:
            fs.log_batch(nonexistent_uuid, [], [], [])
        # assert e.exception.error_code == ErrorCode.Name(RESOURCE_DOES_NOT_EXIST)
        # assert ("Run '%s' not found" % nonexistent_uuid) in e.exception.message

    def test_log_batch_params_idempotency(self):
        fs = self._get_store()
        run = self._create_run(fs)
        params = [Param("p-key", "p-val")]
        fs.log_batch(run.info.run_id, metrics=[], params=params, tags=[])
        fs.log_batch(run.info.run_id, metrics=[], params=params, tags=[])
        self._verify_logged(fs, run.info.run_id, metrics=[], params=params, tags=[])

    def test_log_batch_tags_idempotency(self):
        fs = self._get_store()
        run = self._create_run(fs)
        fs.log_batch(run.info.run_id, metrics=[], params=[], tags=[RunTag("t-key", "t-val")])
        fs.log_batch(run.info.run_id, metrics=[], params=[], tags=[RunTag("t-key", "t-val")])
        self._verify_logged(
            fs, run.info.run_id, metrics=[], params=[], tags=[RunTag("t-key", "t-val")]
        )

    def test_log_batch_allows_tag_overwrite(self):
        fs = self._get_store()
        run = self._create_run(fs)
        fs.log_batch(run.info.run_id, metrics=[], params=[], tags=[RunTag("t-key", "val")])
        fs.log_batch(run.info.run_id, metrics=[], params=[], tags=[RunTag("t-key", "newval")])
        self._verify_logged(
            fs, run.info.run_id, metrics=[], params=[], tags=[RunTag("t-key", "newval")]
        )

    def test_log_batch_same_metric_repeated_single_req(self):
        fs = self._get_store()
        run = self._create_run(fs)
        metric0 = Metric(key="metric-key", value=1, timestamp=2, step=0)
        metric1 = Metric(key="metric-key", value=2, timestamp=3, step=0)
        fs.log_batch(run.info.run_id, params=[], metrics=[metric0, metric1], tags=[])
        self._verify_logged(fs, run.info.run_id, params=[], metrics=[metric0, metric1], tags=[])

    def test_log_batch_same_metric_repeated_multiple_reqs(self):
        fs = self._get_store()
        run = self._create_run(fs)
        metric0 = Metric(key="metric-key", value=1, timestamp=2, step=0)
        metric1 = Metric(key="metric-key", value=2, timestamp=3, step=0)
        fs.log_batch(run.info.run_id, params=[], metrics=[metric0], tags=[])
        self._verify_logged(fs, run.info.run_id, params=[], metrics=[metric0], tags=[])
        fs.log_batch(run.info.run_id, params=[], metrics=[metric1], tags=[])
        self._verify_logged(fs, run.info.run_id, params=[], metrics=[metric0, metric1], tags=[])

    def test_log_batch_allows_tag_overwrite_single_req(self):
        fs = self._get_store()
        run = self._create_run(fs)
        tags = [RunTag("t-key", "val"), RunTag("t-key", "newval")]
        fs.log_batch(run.info.run_id, metrics=[], params=[], tags=tags)
        self._verify_logged(fs, run.info.run_id, metrics=[], params=[], tags=[tags[-1]])

    def test_log_batch_accepts_empty_payload(self):
        fs = self._get_store()
        run = self._create_run(fs)
        fs.log_batch(run.info.run_id, metrics=[], params=[], tags=[])
        self._verify_logged(fs, run.info.run_id, metrics=[], params=[], tags=[])
