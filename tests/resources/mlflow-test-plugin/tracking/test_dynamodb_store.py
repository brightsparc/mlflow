#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import unittest
import uuid

import time

import pytest
import random
import string

from moto.dynamodb2 import mock_dynamodb2

from mlflow.entities.lifecycle_stage import LifecycleStage
from mlflow.entities import Metric, Param, RunStatus, RunTag, ViewType
from mlflow.exceptions import MlflowException
from mlflow.utils.env import get_env

from tracking.dynamodb_store import DynamodbStore

_DYNAMODB_ENDPOINT_URL_VAR = "MLFLOW_DYNAMODB_ENDPOINT_URL"


def _default_endpoint_url():
    return get_env(_DYNAMODB_ENDPOINT_URL_VAR)


def random_int(lo=1, hi=1e10):
    return random.randint(lo, hi)


def random_str(size=10, chars=string.ascii_uppercase + string.digits):
    return "".join(random.choice(chars) for _ in range(size))


class TestDynamodbStore(unittest.TestCase):
    TEST_LOCALHOST = False

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
        for exp in self.experiments:
            # create experiment
            exp_folder = os.path.join(self.table_prefix, exp)
            d = {
                "experiment_id": exp,
                "name": random_str(),
                "artifact_location": exp_folder,
                "lifecycle_stage": LifecycleStage.ACTIVE,  # Must write for tests
            }
            self.exp_data[exp] = d
            self._write_table("experiment", d)
            # add runs
            self.exp_data[exp]["runs"] = []
            for _ in range(run_count):
                run_id = uuid.uuid4().hex
                self.exp_data[exp]["runs"].append(run_id)
                run_info = {
                    "run_uuid": run_id,
                    "run_id": run_id,
                    "experiment_id": exp,
                    "user_id": random_str(random_int(10, 25)),
                    "status": RunStatus.to_string(RunStatus.RUNNING),
                    "start_time": random_int(1, 10),
                    "end_time": random_int(10, 11),
                    "tags": [],
                    "artifact_uri": "/",
                    "lifecycle_stage": LifecycleStage.ACTIVE,  # Must write for tests
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
        experiment_ids = [e.experiment_id for e in fs.list_experiments()]
        for exp_id in self.experiments:
            self.assertTrue(exp_id in experiment_ids)

    def test_get_experiment_by_id(self):
        fs = self._get_store()
        for exp_id in self.experiments:
            exp = fs.get_experiment(exp_id)
            self.assertEqual(exp.experiment_id, exp_id)
            self.assertEqual(exp.name, self.exp_data[exp_id]["name"])
            self.assertEqual(exp.artifact_location, self.exp_data[exp_id]["artifact_location"])

        # test that fake experiments don't exist.
        # look for random experiment ids between 8000, 15000 since created ones are (100, 2000)
        for exp_id in set(random_int(8000, 15000) for x in range(20)):
            with self.assertRaises(Exception):
                fs.get_experiment(str(exp_id))

    def test_get_experiment_by_name(self):
        fs = self._get_store()
        for exp_id in self.experiments:
            name = self.exp_data[exp_id]["name"]
            exp = fs.get_experiment_by_name(name)
            self.assertEqual(exp.experiment_id, exp_id)
            self.assertEqual(exp.name, self.exp_data[exp_id]["name"])
            self.assertEqual(exp.artifact_location, self.exp_data[exp_id]["artifact_location"])

        # test that fake experiments don't exist.
        # look up experiments with names of length 15 since created ones are of length 10
        for exp_names in set(random_str(15) for x in range(20)):
            exp = fs.get_experiment_by_name(exp_names)
            self.assertIsNone(exp)

    def test_create_experiment(self):
        fs = self._get_store()

        # Error cases
        with self.assertRaises(Exception):
            fs.create_experiment(None)
        with self.assertRaises(Exception):
            fs.create_experiment("")

        # Create the experiment
        name = random_str()
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

    def _get_random_experiment_id(self):
        return self.experiments[random_int(0, len(self.experiments) - 1)]

    def _get_random_run_id(self):
        exp_id = self._get_random_experiment_id()
        return self.exp_data[exp_id]["runs"][0]

    def test_delete_restore_experiment(self):
        fs = self._get_store()
        exp_id = self._get_random_experiment_id()
        exp_name = self.exp_data[exp_id]["name"]

        # delete it
        fs.delete_experiment(exp_id)
        self.assertTrue(exp_id not in self._extract_ids(fs.list_experiments(ViewType.ACTIVE_ONLY)))
        self.assertTrue(exp_id in self._extract_ids(fs.list_experiments(ViewType.DELETED_ONLY)))
        experiments = self._extract_ids(fs.list_experiments(ViewType.ALL))
        self.assertTrue(exp_id in experiments)
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
        exp_id = self._get_random_experiment_id()
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
        run_id = self._get_random_run_id()
        # Should not throw.
        run = fs.get_run(run_id)
        assert run.info.lifecycle_stage == "active"
        fs.delete_run(run_id)
        assert fs.get_run(run_id).info.lifecycle_stage == "deleted"
        fs.restore_run(run_id)
        assert fs.get_run(run_id).info.lifecycle_stage == "active"

    def test_create_run_in_deleted_experiment(self):
        fs = self._get_store()
        exp_id = self._get_random_experiment_id()
        # delete it
        fs.delete_experiment(exp_id)
        with pytest.raises(Exception):
            fs.create_run(
                experiment_id=exp_id, user_id="user", start_time=0, tags=[],
            )

    def test_get_run(self):
        fs = self._get_store()
        for exp_id in self.experiments:
            runs = self.exp_data[exp_id]["runs"]
            for run_id in runs:
                run = fs.get_run(run_id)
                print("loaded run", run.info)
                run_info = self.run_data[run_id]
                run_info.pop("metrics")
                run_info.pop("params")
                run_info.pop("tags")
                run_info["lifecycle_stage"] = LifecycleStage.ACTIVE
                self.assertEqual(run_info, dict(run.info))

    def test_list_run_infos(self):
        fs = self._get_store()
        for exp_id in self.experiments:
            run_infos = fs.list_run_infos(exp_id, run_view_type=ViewType.ALL)
            for run_info in run_infos:
                run_id = run_info.run_id
                dict_run_info = self.run_data[run_id]
                #  In some cases metrics might be missing
                dict_run_info.pop("metrics", None)
                dict_run_info.pop("params", None)
                dict_run_info.pop("tags", None)
                dict_run_info["lifecycle_stage"] = LifecycleStage.ACTIVE
                self.assertEqual(dict_run_info, dict(run_info))

    def test_get_metric(self):
        fs = self._get_store()
        for exp_id in self.experiments:
            runs = self.exp_data[exp_id]["runs"]
            for run_id in runs:
                run_info = self.run_data[run_id]
                metrics_dict = run_info.pop("metrics")
                for metric_name, values in metrics_dict.items():
                    # just the last recorded value
                    timestamp, metric_value = values[-1]
                    metric = fs.get_metric(run_id, metric_name)
                    self.assertEqual(metric.timestamp, timestamp)
                    self.assertEqual(metric.key, metric_name)
                    self.assertEqual(metric.value, metric_value)

    def test_get_all_metrics(self):
        fs = self._get_store()
        for exp_id in self.experiments:
            runs = self.exp_data[exp_id]["runs"]
            for run_id in runs:
                run_info = self.run_data[run_id]
                metrics = fs.get_all_metrics(run_id)
                metrics_dict = run_info.pop("metrics")
                for metric in metrics:
                    # just the last recorded value
                    timestamp, metric_value = metrics_dict[metric.key][-1]
                    self.assertEqual(metric.timestamp, timestamp)
                    self.assertEqual(metric.value, metric_value)

    def test_get_metric_history(self):
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

    def test_get_param(self):
        fs = self._get_store()
        for exp_id in self.experiments:
            runs = self.exp_data[exp_id]["runs"]
            for run_id in runs:
                run_info = self.run_data[run_id]
                params_dict = run_info.pop("params")
                for param_name, param_value in params_dict.items():
                    param = fs.get_param(run_id, param_name)
                    self.assertEqual(param.key, param_name)
                    self.assertEqual(param.value, param_value)

    def test_search_runs(self):
        # replace with test with code is implemented
        fs = self._get_store()
        # Expect 2 runs for each experiment
        exp_id = self._get_random_experiment_id()
        runs = fs.search_runs([exp_id], None, run_view_type=ViewType.ACTIVE_ONLY)
        assert len(runs) == 2
        runs = fs.search_runs([exp_id], None, run_view_type=ViewType.ALL)
        assert len(runs) == 2
        runs = fs.search_runs([exp_id], None, run_view_type=ViewType.DELETED_ONLY)
        assert len(runs) == 0

    def test_weird_param_names(self):
        WEIRD_PARAM_NAME = "this is/a weird/but valid param"
        fs = self._get_store()
        run_id = self._get_random_run_id()
        fs.log_param(run_id, Param(WEIRD_PARAM_NAME, "Value"))
        param = fs.get_param(run_id, WEIRD_PARAM_NAME)
        assert param.key == WEIRD_PARAM_NAME
        assert param.value == "Value"

    def test_weird_metric_names(self):
        # moto doesn't support list_append see: https://github.com/spulec/moto/issues/847
        if self.endpoint_url:
            WEIRD_METRIC_NAME = "this is/a weird/but valid metric"
            fs = self._get_store()
            run_id = self._get_random_run_id()
            fs.log_metric(run_id, Metric(WEIRD_METRIC_NAME, 10, 1234, 0))
            metric = fs.get_metric(run_id, WEIRD_METRIC_NAME)
            assert metric.key == WEIRD_METRIC_NAME
            assert metric.value == 10
            assert metric.timestamp == 1234

    def test_weird_tag_names(self):
        WEIRD_TAG_NAME = "this is/a weird/but valid tag"
        WEIRD_TAG_VALUE = "some value"
        fs = self._get_store()
        run_id = self._get_random_run_id()
        fs.set_tag(run_id, RunTag(WEIRD_TAG_NAME, WEIRD_TAG_VALUE))
        tags = fs.get_run(run_id).data.tags
        assert WEIRD_TAG_NAME in tags
        assert tags[WEIRD_TAG_NAME] == WEIRD_TAG_VALUE

    def test_set_tags(self):
        fs = self._get_store()
        run_id = self._get_random_run_id()
        fs.set_tag(run_id, RunTag("tag0", "value0"))
        fs.set_tag(run_id, RunTag("tag1", "value1"))
        tags = fs.get_run(run_id).data.tags
        assert tags == {
            "tag0": "value0",
            "tag1": "value1",
        }

        # Can overwrite tags.
        fs.set_tag(run_id, RunTag("tag0", "value2"))
        tags = fs.get_run(run_id).data.tags
        assert tags == {
            "tag0": "value2",
            "tag1": "value1",
        }

        # Can set multiline tags.
        fs.set_tag(run_id, RunTag("multiline_tag", "value2\nvalue2\nvalue2"))
        tags = fs.get_run(run_id).data.tags
        assert tags == {
            "tag0": "value2",
            "tag1": "value1",
            "multiline_tag": "value2\nvalue2\nvalue2",
        }

    def test_unicode_tag(self):
        fs = self._get_store()
        run_id = self._get_random_run_id()
        value = u"𝐼 𝓈𝑜𝓁𝑒𝓂𝓃𝓁𝓎 𝓈𝓌𝑒𝒶𝓇 𝓉𝒽𝒶𝓉 𝐼 𝒶𝓂 𝓊𝓅 𝓉𝑜 𝓃𝑜 𝑔𝑜𝑜𝒹"
        fs.set_tag(run_id, RunTag("message", value))
        tags = fs.get_run(run_id).data.tags
        assert "message" in tags
        assert tags["message"] == value

    def test_get_deleted_run(self):
        """
        Getting metrics/tags/params/run info should be allowed on deleted runs.
        """
        fs = self._get_store()
        run_id = self._get_random_run_id()
        fs.delete_run(run_id)

        run = fs.get_run(run_id)
        metric_key = list(run.data.metrics)[0]
        assert fs.get_metric(run_id, metric_key).value == run.data.metrics[metric_key]
        params_key = list(run.data.params)[0]
        assert fs.get_param(run_id, params_key).value == run.data.params[params_key]

    def test_set_deleted_run(self):
        """
        Setting metrics/tags/params/updating run info should not be allowed on deleted runs.
        """
        fs = self._get_store()
        run_id = self._get_random_run_id()
        fs.delete_run(run_id)

        assert fs.get_run(run_id).info.lifecycle_stage == LifecycleStage.DELETED
        with pytest.raises(MlflowException):
            fs.set_tag(run_id, RunTag("a", "b"))
        with pytest.raises(MlflowException):
            fs.log_metric(run_id, Metric("a", value=0.0, timestamp=0, step=0))
        with pytest.raises(MlflowException):
            fs.log_param(run_id, Param("a", "b"))

    def test_default_experiment_initialization(self):
        fs = self._get_store()
        exp_id = self._get_random_experiment_id()
        fs.delete_experiment(exp_id)
        fs = self._get_store()
        assert fs.get_experiment(exp_id).lifecycle_stage == LifecycleStage.DELETED
