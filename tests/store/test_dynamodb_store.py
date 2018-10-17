#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import shutil
import unittest
import uuid

import time

import pytest

import boto3
from mock import Mock
from moto.dynamodb2 import mock_dynamodb2

from mlflow.entities import Experiment, Metric, Param, RunTag, ViewType, RunInfo
from mlflow.exceptions import MlflowException
from mlflow.store.file_store import FileStore
from mlflow.store.dynamodb_store import DynamodbStore
from mlflow.utils.mlflow_tags import MLFLOW_PARENT_RUN_ID
from tests.helper_functions import random_int, random_str

class TestDynamodbStore(unittest.TestCase):
    def setUp(self):
        self.mock = mock_dynamodb2()
        self.mock.start()

        self.table_prefix='mlflow'
        print('creeate tables')
        self._create_tables()
        print('populate tables')
        self._populate_tables(exp_count=1, param_count=1, metric_count=1, values_count=3)
        self.maxDiff = None

    def _create_tables(self):
        # create a mock dynamodb client, and create tables
        client = boto3.client('dynamodb')
        response = client.create_table(
            AttributeDefinitions=[
                {
                    'AttributeName': 'experiment_id',
                    'AttributeType': 'N'
                },
                {
                    'AttributeName': 'lifecycle_stage',
                    'AttributeType': 'S'
                },

                {
                    'AttributeName': 'name',
                    'AttributeType': 'S'
                },

            ],
            TableName='{}_experiment'.format(self.table_prefix),
            KeySchema=[
                {
                    'AttributeName': 'experiment_id',
                    'KeyType': 'HASH'
                },
            ],
            GlobalSecondaryIndexes=[
                {
                    'IndexName': 'LifeCycleStage',
                    'KeySchema': [
                        {
                            'AttributeName': 'lifecycle_stage',
                            'KeyType': 'HASH'
                        },
                        {
                            'AttributeName': 'name',
                            'KeyType': 'RANGE'
                        },

                    ],
                    'Projection': {
                        'ProjectionType': 'KEYS_ONLY'
                    },
                    'ProvisionedThroughput': {
                        'ReadCapacityUnits': 1,
                        'WriteCapacityUnits': 1
                    }
                },
            ],
            ProvisionedThroughput={
                'ReadCapacityUnits': 1,
                'WriteCapacityUnits': 1
            },
        )
        print('create table experiment')
        response = client.create_table(
            AttributeDefinitions=[
                {
                    'AttributeName': 'run_uuid',
                    'AttributeType': 'S'
                },
                {
                    'AttributeName': 'experiment_id',
                    'AttributeType': 'N'
                },
                {
                    'AttributeName': 'lifecycle_stage',
                    'AttributeType': 'S'
                },

            ],
            TableName='{}_run'.format(self.table_prefix),
            KeySchema=[
                {
                    'AttributeName': 'run_uuid',
                    'KeyType': 'HASH'
                },
            ],
            GlobalSecondaryIndexes=[
                {
                    'IndexName': 'LifeCycleStage',
                    'KeySchema': [
                        {
                            'AttributeName': 'lifecycle_stage',
                            'KeyType': 'HASH'
                        },
                        {
                            'AttributeName': 'experiment_id',
                            'KeyType': 'RANGE'
                        },

                    ],
                    'Projection': {
                        'ProjectionType': 'KEYS_ONLY'
                    },
                    'ProvisionedThroughput': {
                        'ReadCapacityUnits': 1,
                        'WriteCapacityUnits': 1
                    }
                },
            ],
            ProvisionedThroughput={
                'ReadCapacityUnits': 1,
                'WriteCapacityUnits': 1
            },
        )
        print('create table run')
        for key in ['tag', 'param', 'metric']:
            table_name = '{}_run_{}'.format(self.table_prefix, key)
            response = client.create_table(
                AttributeDefinitions=[
                    {
                        'AttributeName': 'run_uuid',
                        'AttributeType': 'S'
                    },
                    {
                        'AttributeName': 'key',
                        'AttributeType': 'S'
                    },

                ],
                TableName=table_name,
                KeySchema=[
                    {
                        'AttributeName': 'run_uuid',
                        'KeyType': 'HASH'
                    },
                    {
                        'AttributeName': 'key',
                        'KeyType': 'RANGE'
                    },

                ],
                ProvisionedThroughput={
                    'ReadCapacityUnits': 1,
                    'WriteCapacityUnits': 1
                },
            )
            print('create table', table_name)

    def _write_table(self, name, d):
        # Use mock dnamodb to put to table
        dynamodb = boto3.resource('dynamodb')
        table_name = '_'.join([self.table_prefix,name])
        table = dynamodb.Table(table_name)
        response = table.put_item(Item=d)
        print('write table', name)

    def _populate_tables(self, exp_count=3, run_count=2, param_count=5, metric_count=3, values_count=10):
        self.experiments = [random_int(100, int(1e9)) for _ in range(exp_count)]
        self.exp_data = {}
        self.run_data = {}
        # Include default experiment
        #self.experiments.append(Experiment.DEFAULT_EXPERIMENT_ID)
        self.experiments[0] = Experiment.DEFAULT_EXPERIMENT_ID # why do we need default?
        for exp in self.experiments:
            # create experiment
            exp_folder = os.path.join(self.table_prefix, str(exp))
            d = {"experiment_id": exp, "name": random_str(), "artifact_location": exp_folder}
            self.exp_data[exp] = d
            self._write_table('experiment', d)
            # add runs
            self.exp_data[exp]["runs"] = []
            for _ in range(run_count):
                run_uuid = uuid.uuid4().hex
                self.exp_data[exp]["runs"].append(run_uuid)
                run_folder = os.path.join(exp_folder, run_uuid)
                run_info = {"run_uuid": run_uuid,
                            "experiment_id": exp,
                            "name": random_str(random_int(10, 40)), # reserved word?
                            "source_type": random_int(1, 4),
                            "source_name": random_str(random_int(100, 300)),
                            "entry_point_name": random_str(random_int(100, 300)),
                            "user_id": random_str(random_int(10, 25)),
                            "status": random_int(1, 5),
                            "start_time": random_int(1, 10),
                            "end_time": random_int(20, 30),
                            "source_version": random_str(random_int(10, 30)),
                            "tags": [],
                            "artifact_uri": "%s/%s" % (run_folder, FileStore.ARTIFACTS_FOLDER_NAME),
                            "lifecycle_stage": RunInfo.ACTIVE_LIFECYCLE, # do we need to write this?
                            }
                self._write_table('run', run_info)
                self.run_data[run_uuid] = run_info
                # params
                params_folder = os.path.join(run_folder, FileStore.PARAMS_FOLDER_NAME)
                params = {}
                for _ in range(param_count):
                    param_name = random_str(random_int(4, 12))
                    param_value = random_str(random_int(10, 15))
                    self._write_table('run_param', {
                        'run_uuid': run_uuid,
                        'key':  param_name,
                        'value': param_value
                    })
                    params[param_name] = param_value
                self.run_data[run_uuid]["params"] = params
                # metrics
                metrics_folder = os.path.join(run_folder, FileStore.METRICS_FOLDER_NAME)
                metrics = {}
                for _ in range(metric_count):
                    metric_name = random_str(random_int(6, 10))
                    timestamp = int(time.time())
                    metric_file = os.path.join(metrics_folder, metric_name)
                    values, values_map = [], []
                    for _ in range(values_count):
                        metric_value = random_int(100, 2000)
                        timestamp += random_int(10000, 2000000)
                        values.append((timestamp, metric_value))
                        values_map.insert(0, { 'timestamp': timestamp, 'value':  metric_value })
                    self._write_table('run_metric', {
                        'run_uuid': run_uuid,
                        'key':  metric_name,
                        'metrics': values_map
                    })
                    metrics[metric_name] = values # Tweak results for key/value instead of index 0/1
                self.run_data[run_uuid]["metrics"] = metrics

    def tearDown(self):
        self.mock.stop()

    def test_list_experiments(self):
        fs = DynamodbStore(self.table_prefix)
        for exp in fs.list_experiments():
            exp_id = exp.experiment_id
            self.assertTrue(exp_id in self.experiments)
            self.assertEqual(exp.name, self.exp_data[exp_id]["name"])
            self.assertEqual(exp.artifact_location, self.exp_data[exp_id]["artifact_location"])

    def test_get_experiment_by_id(self):
        fs = DynamodbStore(self.table_prefix)
        for exp_id in self.experiments:
            exp = fs.get_experiment(exp_id)
            self.assertEqual(exp.experiment_id, exp_id)
            self.assertEqual(exp.name, self.exp_data[exp_id]["name"])
            self.assertEqual(exp.artifact_location, self.exp_data[exp_id]["artifact_location"])

        # test that fake experiments dont exist.
        # look for random experiment ids between 8000, 15000 since created ones are (100, 2000)
        for exp_id in set(random_int(8000, 15000) for x in range(20)):
            with self.assertRaises(Exception):
                fs.get_experiment(exp_id)

    def test_get_experiment_by_name(self):
        fs = DynamodbStore(self.table_prefix)
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

    def test_create_experiment(self):
        fs = DynamodbStore(self.table_prefix)

        # Error cases
        with self.assertRaises(Exception):
            fs.create_experiment(None)
        with self.assertRaises(Exception):
            fs.create_experiment("")

        next_id = max(self.experiments) + 1
        name = random_str(25)  # since existing experiments are 10 chars long
        created_id = fs.create_experiment(name)
        # test that newly created experiment matches expected id
        self.assertEqual(created_id, next_id)

        # get the new experiment (by id) and verify (by name)
        exp1 = fs.get_experiment(created_id)
        self.assertEqual(exp1.name, name)

        # get the new experiment (by name) and verify (by id)
        exp2 = fs.get_experiment_by_name(name)
        self.assertEqual(exp2.experiment_id, created_id)

    def test_create_duplicate_experiments(self):
        fs = DynamodbStore(self.table_prefix)
        for exp_id in self.experiments:
            name = self.exp_data[exp_id]["name"]
            with self.assertRaises(Exception):
                fs.create_experiment(name)

    def _extract_ids(self, experiments):
        return [e.experiment_id for e in experiments]

    def test_delete_restore_experiment(self):
        fs = DynamodbStore(self.table_prefix)
        exp_id = self.experiments[random_int(0, len(self.experiments) - 1)]
        exp_name = self.exp_data[exp_id]["name"]

        # delete it
        fs.delete_experiment(exp_id)
        self.assertTrue(exp_id not in self._extract_ids(fs.list_experiments(ViewType.ACTIVE_ONLY)))
        self.assertTrue(exp_id in self._extract_ids(fs.list_experiments(ViewType.DELETED_ONLY)))
        self.assertTrue(exp_id in self._extract_ids(fs.list_experiments(ViewType.ALL)))
        self.assertEqual(fs.get_experiment(exp_id).lifecycle_stage,
                         Experiment.DELETED_LIFECYCLE)

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
        self.assertEqual(fs.get_experiment(exp_id).lifecycle_stage,
                         Experiment.ACTIVE_LIFECYCLE)

    def test_rename_experiment(self):
        fs = DynamodbStore(self.table_prefix)
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
        assert 'non-active lifecycle' in str(e.value)
        self.assertEqual(fs.get_experiment(exp_id).name, new_name)

        # Restore the experiment, and confirm that we acn now rename it.
        fs.restore_experiment(exp_id)
        self.assertEqual(fs.get_experiment(exp_id).name, new_name)
        fs.rename_experiment(exp_id, exp_name)
        self.assertEqual(fs.get_experiment(exp_id).name, exp_name)

    def test_delete_restore_run(self):
        fs = DynamodbStore(self.table_prefix)
        exp_id = self.experiments[random_int(0, len(self.experiments) - 1)]
        run_id = self.exp_data[exp_id]['runs'][0]
        # Should not throw.
        run = fs.get_run(run_id)
        assert run.info.lifecycle_stage == 'active'
        fs.delete_run(run_id)
        assert fs.get_run(run_id).info.lifecycle_stage == 'deleted'
        fs.restore_run(run_id)
        assert fs.get_run(run_id).info.lifecycle_stage == 'active'

    def test_create_run_in_deleted_experiment(self):
        fs = DynamodbStore(self.table_prefix)
        exp_id = self.experiments[random_int(0, len(self.experiments) - 1)]
        # delete it
        fs.delete_experiment(exp_id)
        with pytest.raises(Exception):
            source_type = 1
            fs.create_run(exp_id, 'user', 'name', source_type, 'source_name', 'entry_point_name',
                          0, None, [], None)

    def test_get_run(self):
        fs = DynamodbStore(self.table_prefix)
        for exp_id in self.experiments:
            runs = self.exp_data[exp_id]["runs"]
            for run_uuid in runs:
                run = fs.get_run(run_uuid)
                run_info = self.run_data[run_uuid]
                run_info.pop("metrics")
                run_info.pop("params")
                run_info.pop("tags")
                run_info['lifecycle_stage'] = RunInfo.ACTIVE_LIFECYCLE
                self.assertEqual(run_info, dict(run.info))

    def test_list_run_infos(self):
        fs = DynamodbStore(self.table_prefix)
        for exp_id in self.experiments:
            run_infos = fs.list_run_infos(exp_id, run_view_type=ViewType.ALL)
            for run_info in run_infos:
                run_uuid = run_info.run_uuid
                dict_run_info = self.run_data[run_uuid]
                dict_run_info.pop("metrics")
                dict_run_info.pop("params")
                dict_run_info.pop("tags")
                dict_run_info['lifecycle_stage'] = RunInfo.ACTIVE_LIFECYCLE
                self.assertEqual(dict_run_info, dict(run_info))

    def test_get_metric(self):
        fs = DynamodbStore(self.table_prefix)
        for exp_id in self.experiments:
            runs = self.exp_data[exp_id]["runs"]
            for run_uuid in runs:
                run_info = self.run_data[run_uuid]
                metrics_dict = run_info.pop("metrics")
                for metric_name, values in metrics_dict.items():
                    # just the last recorded value
                    timestamp, metric_value = values[-1]
                    metric = fs.get_metric(run_uuid, metric_name)
                    self.assertEqual(metric.timestamp, timestamp)
                    self.assertEqual(metric.key, metric_name)
                    self.assertEqual(metric.value, metric_value)

    def test_get_all_metrics(self):
        fs = DynamodbStore(self.table_prefix)
        for exp_id in self.experiments:
            runs = self.exp_data[exp_id]["runs"]
            for run_uuid in runs:
                run_info = self.run_data[run_uuid]
                metrics = fs.get_all_metrics(run_uuid)
                metrics_dict = run_info.pop("metrics")
                for metric in metrics:
                    # just the last recorded value
                    timestamp, metric_value = metrics_dict[metric.key][-1]
                    self.assertEqual(metric.timestamp, timestamp)
                    self.assertEqual(metric.value, metric_value)

    def test_get_metric_history(self):
        fs = DynamodbStore(self.table_prefix)
        for exp_id in self.experiments:
            runs = self.exp_data[exp_id]["runs"]
            for run_uuid in runs:
                run_info = self.run_data[run_uuid]
                metrics = run_info.pop("metrics")
                for metric_name, values in metrics.items():
                    metric_history = fs.get_metric_history(run_uuid, metric_name)
                    sorted_values = sorted(values, reverse=True)
                    for metric in metric_history:
                        timestamp, metric_value = sorted_values.pop()
                        self.assertEqual(metric.timestamp, timestamp)
                        self.assertEqual(metric.key, metric_name)
                        self.assertEqual(metric.value, metric_value)

    def test_get_param(self):
        fs = DynamodbStore(self.table_prefix)
        for exp_id in self.experiments:
            runs = self.exp_data[exp_id]["runs"]
            for run_uuid in runs:
                run_info = self.run_data[run_uuid]
                params_dict = run_info.pop("params")
                for param_name, param_value in params_dict.items():
                    param = fs.get_param(run_uuid, param_name)
                    self.assertEqual(param.key, param_name)
                    self.assertEqual(param.value, param_value)

    def test_search_runs(self):
        # replace with test with code is implemented
        fs = DynamodbStore(self.table_prefix)
        # Expect 2 runs for each experiment
        runs = fs.search_runs([self.experiments[0]], [], run_view_type=ViewType.ACTIVE_ONLY)
        assert len(runs) == 2
        runs = fs.search_runs([self.experiments[0]], [], run_view_type=ViewType.ALL)
        assert len(runs) == 2
        runs = fs.search_runs([self.experiments[0]], [], run_view_type=ViewType.DELETED_ONLY)
        assert len(runs) == 0

    def test_weird_param_names(self):
        WEIRD_PARAM_NAME = "this is/a weird/but valid param"
        fs = DynamodbStore(self.table_prefix)
        run_uuid = self.exp_data[0]["runs"][0]
        fs.log_param(run_uuid, Param(WEIRD_PARAM_NAME, "Value"))
        param = fs.get_param(run_uuid, WEIRD_PARAM_NAME)
        assert param.key == WEIRD_PARAM_NAME
        assert param.value == "Value"

    # def test_weird_metric_names(self):
    #     WEIRD_METRIC_NAME = "this is/a weird/but valid metric"
    #     fs = DynamodbStore(self.table_prefix)
    #     run_uuid = self.exp_data[0]["runs"][0]
    #     fs.log_metric(run_uuid, Metric(WEIRD_METRIC_NAME, 10, 1234))
    #     metric = fs.get_metric(run_uuid, WEIRD_METRIC_NAME)
    #     assert metric.key == WEIRD_METRIC_NAME
    #     assert metric.value == 10
    #     assert metric.timestamp == 1234

    def test_weird_tag_names(self):
        WEIRD_TAG_NAME = "this is/a weird/but valid tag"
        fs = DynamodbStore(self.table_prefix)
        run_uuid = self.exp_data[0]["runs"][0]
        fs.set_tag(run_uuid, RunTag(WEIRD_TAG_NAME, "Muhahaha!"))
        tag = fs.get_run(run_uuid).data.tags[0]
        assert tag.key == WEIRD_TAG_NAME
        assert tag.value == "Muhahaha!"

    def test_set_tags(self):
        fs = DynamodbStore(self.table_prefix)
        run_uuid = self.exp_data[0]["runs"][0]
        fs.set_tag(run_uuid, RunTag("tag0", "value0"))
        fs.set_tag(run_uuid, RunTag("tag1", "value1"))
        tags = [(t.key, t.value) for t in fs.get_run(run_uuid).data.tags]
        assert set(tags) == {
            ("tag0", "value0"),
            ("tag1", "value1"),
        }

        # Can overwrite tags.
        fs.set_tag(run_uuid, RunTag("tag0", "value2"))
        tags = [(t.key, t.value) for t in fs.get_run(run_uuid).data.tags]
        assert set(tags) == {
            ("tag0", "value2"),
            ("tag1", "value1"),
        }

        # Can set multiline tags.
        fs.set_tag(run_uuid, RunTag("multiline_tag", "value2\nvalue2\nvalue2"))
        tags = [(t.key, t.value) for t in fs.get_run(run_uuid).data.tags]
        assert set(tags) == {
            ("tag0", "value2"),
            ("tag1", "value1"),
            ("multiline_tag", "value2\nvalue2\nvalue2"),
        }

    def test_unicode_tag(self):
        fs = DynamodbStore(self.table_prefix)
        run_uuid = self.exp_data[0]["runs"][0]
        value = u"𝐼 𝓈𝑜𝓁𝑒𝓂𝓃𝓁𝓎 𝓈𝓌𝑒𝒶𝓇 𝓉𝒽𝒶𝓉 𝐼 𝒶𝓂 𝓊𝓅 𝓉𝑜 𝓃𝑜 𝑔𝑜𝑜𝒹"
        fs.set_tag(run_uuid, RunTag("message", value))
        tag = fs.get_run(run_uuid).data.tags[0]
        assert tag.key == "message"
        assert tag.value == value

    def test_get_deleted_run(self):
        """
        Getting metrics/tags/params/run info should be allowed on deleted runs.
        """
        fs = DynamodbStore(self.table_prefix)
        exp_id = self.experiments[random_int(0, len(self.experiments) - 1)]
        run_id = self.exp_data[exp_id]['runs'][0]
        fs.delete_run(run_id)

        run = fs.get_run(run_id)
        assert fs.get_metric(run_id, run.data.metrics[0].key).value == run.data.metrics[0].value
        assert fs.get_param(run_id, run.data.params[0].key).value == run.data.params[0].value

    def test_set_deleted_run(self):
        """
        Setting metrics/tags/params/updating run info should not be allowed on deleted runs.
        """
        fs = DynamodbStore(self.table_prefix)
        exp_id = self.experiments[random_int(0, len(self.experiments) - 1)]
        run_id = self.exp_data[exp_id]['runs'][0]
        fs.delete_run(run_id)

        assert fs.get_run(run_id).info.lifecycle_stage == RunInfo.DELETED_LIFECYCLE
        with pytest.raises(MlflowException):
            fs.set_tag(run_id, RunTag('a', 'b'))
        with pytest.raises(MlflowException):
            fs.log_metric(run_id, Metric('a', 0.0, timestamp=0))
        with pytest.raises(MlflowException):
            fs.log_param(run_id, Param('a', 'b'))

    def test_create_run_with_parent_id(self):
        fs = DynamodbStore(self.table_prefix)
        exp_id = self.experiments[random_int(0, len(self.experiments) - 1)]
        source_type = 1
        run = fs.create_run(exp_id, 'user', 'name', source_type, 'source_name',
                            'entry_point_name', 0, None, [], 'test_parent_run_id')
        assert any([t.key == MLFLOW_PARENT_RUN_ID and t.value == 'test_parent_run_id'
                    for t in fs.get_all_tags(run.info.run_uuid)])

    def test_default_experiment_initialization(self):
        fs = DynamodbStore(self.table_prefix)
        fs.delete_experiment(Experiment.DEFAULT_EXPERIMENT_ID)
        fs = DynamodbStore(self.table_prefix)
        assert fs.get_experiment(0).lifecycle_stage == Experiment.DELETED_LIFECYCLE
