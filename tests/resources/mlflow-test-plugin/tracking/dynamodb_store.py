import os
import uuid
from six.moves import urllib

import botocore
import boto3
from boto3.dynamodb.conditions import Key, And
from decimal import Decimal

from mlflow.entities.lifecycle_stage import LifecycleStage
from mlflow.entities import (
    Experiment,
    Metric,
    Param,
    Run,
    RunData,
    RunInfo,
    RunStatus,
    RunTag,
    ViewType,
)
from mlflow.entities.run_info import check_run_is_active, check_run_is_deleted
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import (
    INVALID_PARAMETER_VALUE,
    RESOURCE_ALREADY_EXISTS,
    INVALID_STATE,
    RESOURCE_DOES_NOT_EXIST,
    INTERNAL_ERROR,
)
from mlflow.store.tracking.abstract_store import AbstractStore
from mlflow.utils.validation import (
    _validate_metric_name,
    _validate_param_name,
    _validate_run_id,
    _validate_tag_name,
    _validate_batch_log_limits,
    _validate_batch_log_data,
)

from mlflow.utils.env import get_env
from mlflow.utils.search_utils import SearchUtils

_DYNAMODB_ENDPOINT_URL_VAR = "MLFLOW_DYNAMODB_ENDPOINT_URL"
_DYNAMODB_TABLE_PREFIX_VAR = "MLFLOW_DYNAMODB_TABLE_PREFIX"


def _default_endpoint_url():
    return get_env(_DYNAMODB_ENDPOINT_URL_VAR)


def _default_table_prefix():
    return get_env(_DYNAMODB_TABLE_PREFIX_VAR) or "mlflow"


def _dict_to_experiment(d):
    return Experiment(
        experiment_id=str(d["experiment_id"]),
        name=d["name"],
        artifact_location=d.get("artifact_location") or "",
        lifecycle_stage=d.get("lifecycle_stage", LifecycleStage.ACTIVE),
        tags=d.get("tags") or [],  # Store tags directly against experiment
    )


def _dict_to_run_info(d):
    return RunInfo(
        run_id=d["run_id"],
        run_uuid=d["run_id"],
        experiment_id=str(d["experiment_id"]),
        user_id=d.get("user_id") or "",
        status=d.get("status") or RunStatus.to_string(RunStatus.RUNNING),
        start_time=int(d.get("start_time") or 0),
        end_time=int(d.get("end_time") or 0),
        lifecycle_stage=d.get("lifecycle_stage", LifecycleStage.ACTIVE),
        artifact_uri=d.get("artifact_uri") or "",
    )


def _list_to_run_tag(l):
    return [RunTag(key=d["key"], value=d["value"]) for d in l]


def _list_to_run_param(l):
    return [Param(key=d["key"], value=d["value"]) for d in l]


# Return the first element in the list, which is the most recent if more than one.
def _list_to_run_metric(l):
    return [
        Metric(
            key=rm["key"],
            value=float(rm["metrics"][0]["value"]),
            timestamp=int(rm["metrics"][0]["timestamp"]),
            step=0,
        )
        for rm in l
    ]


# Return the metrics with most recent at the end of the list
def _dict_to_run_metric_history(rm):
    return [
        Metric(key=rm["key"], value=float(m["value"]), timestamp=int(m["timestamp"]), step=i,)
        for (i, m) in enumerate(rm["metrics"][::-1])
    ]


def _entity_to_dict(obj):
    return {k: None if v == "" else v for k, v in dict(obj).items()}


def _filter_view_type(d, view_type=None):
    return (
        view_type is None
        or view_type == ViewType.ALL
        or (view_type == ViewType.ACTIVE_ONLY and d.get("lifecycle_stage") == LifecycleStage.ACTIVE)
        or (
            view_type == ViewType.DELETED_ONLY
            and d.get("lifecycle_stage") == LifecycleStage.DELETED
        )
    )


def _filter_experiment(experiments, view_type=None, name=None):
    return [
        e
        for e in experiments
        if _filter_view_type(e, view_type) and (name is None or e["name"] == name)
    ]


def _filter_run(runs, view_type=None, experiment_id=None):
    return [
        r
        for r in runs
        if _filter_view_type(r, view_type)
        and (experiment_id is None or r["experiment_id"] == experiment_id)
    ]


class DynamodbStore(AbstractStore):
    EXPERIMENT_TABLE = "experiment"
    RUN_TABLE = "run"
    METRICS_TABLE = "run_metric"
    PARAMS_TABLE = "run_param"
    TAGS_TABLE = "run_tag"

    def __init__(
        self,
        store_uri=None,
        artifact_uri=None,  # Not supported by must be included
        endpoint_url=None,
        region_name=None,
        use_gsi=True,
        use_projections=True,
        create_tables=True,
    ):
        """
        Create a new DynamodbStore for storing experiments and runs.

        :param store_uri: DynamoDb scheme followed by table name 'dynamodb:mlflow'
        :param artifact_uri: Required artifacts scheme
        :param endpoint_url: Optional endpoint url for testing Dynamodb Local
        :param region_name: Optional Name of the AWS region for Dynamodb
        :param use_gsi: Flag to query Global Secondary Indices, defaults to True.
        :param use_projections: Flag to use projections in queries, defaults to True.
        """
        super(DynamodbStore, self).__init__()
        table_prefix = urllib.parse.urlparse(store_uri).path if store_uri else None
        self.table_prefix = table_prefix or _default_table_prefix()
        self.endpoint_url = endpoint_url or _default_endpoint_url()
        self.region_name = region_name
        self.use_gsi = use_gsi
        self.use_projections = use_projections
        # Create tables if they don't exists along with default experiment (ID=0)
        if create_tables and not self.check_tables_exist():
            self.create_tables()
            self.create_experiment("Default")

    def _get_dynamodb_client(self):
        return boto3.client(
            "dynamodb", endpoint_url=self.endpoint_url, region_name=self.region_name
        )

    def _get_dynamodb_resource(self):
        return boto3.resource(
            "dynamodb", endpoint_url=self.endpoint_url, region_name=self.region_name
        )

    def check_tables_exist(self):
        try:
            client = self._get_dynamodb_client()
            table_name = "{}_{}".format(self.table_prefix, DynamodbStore.EXPERIMENT_TABLE)
            response = client.describe_table(TableName=table_name)
            return response["Table"] is None
        except botocore.exceptions.ClientError as error:
            print("error describing table", error)
            return False
        except Exception as error:
            raise error

    def create_tables(self, rcu=1, wcu=1):
        print("creating tables...")
        client = self._get_dynamodb_client()
        waiter = client.get_waiter("table_exists")

        table_name = "{}_{}".format(self.table_prefix, DynamodbStore.EXPERIMENT_TABLE)
        response = client.create_table(
            AttributeDefinitions=[
                {"AttributeName": "experiment_id", "AttributeType": "S"},
                {"AttributeName": "lifecycle_stage", "AttributeType": "S"},
                {"AttributeName": "name", "AttributeType": "S"},
            ],
            TableName=table_name,
            KeySchema=[{"AttributeName": "experiment_id", "KeyType": "HASH"}],
            GlobalSecondaryIndexes=[
                {
                    "IndexName": "LifeCycleStage",
                    "KeySchema": [
                        {"AttributeName": "lifecycle_stage", "KeyType": "HASH"},
                        {"AttributeName": "name", "KeyType": "RANGE"},
                    ],
                    "Projection": {"ProjectionType": "ALL"},
                    "ProvisionedThroughput": {"ReadCapacityUnits": rcu, "WriteCapacityUnits": wcu},
                },
            ],
            ProvisionedThroughput={"ReadCapacityUnits": rcu, "WriteCapacityUnits": wcu},
        )
        if response["ResponseMetadata"]["HTTPStatusCode"] != 200:
            raise MlflowException("Unable to create table '%s'" % table_name)
        waiter.wait(TableName=table_name)
        print("table {} created".format(table_name))

        table_name = "{}_{}".format(self.table_prefix, DynamodbStore.RUN_TABLE)
        response = client.create_table(
            AttributeDefinitions=[
                {"AttributeName": "run_id", "AttributeType": "S"},
                {"AttributeName": "experiment_id", "AttributeType": "S"},
                {"AttributeName": "lifecycle_stage", "AttributeType": "S"},
            ],
            TableName=table_name,
            KeySchema=[{"AttributeName": "run_id", "KeyType": "HASH"}],
            GlobalSecondaryIndexes=[
                {
                    "IndexName": "LifeCycleStage",
                    "KeySchema": [
                        {"AttributeName": "lifecycle_stage", "KeyType": "HASH"},
                        {"AttributeName": "experiment_id", "KeyType": "RANGE"},
                    ],
                    "Projection": {"ProjectionType": "KEYS_ONLY"},
                    "ProvisionedThroughput": {"ReadCapacityUnits": rcu, "WriteCapacityUnits": wcu},
                },
            ],
            ProvisionedThroughput={"ReadCapacityUnits": rcu, "WriteCapacityUnits": wcu},
        )
        if response["ResponseMetadata"]["HTTPStatusCode"] != 200:
            raise MlflowException("Unable to create table '%s'" % table_name)
        waiter.wait(TableName=table_name)
        print("table {} created".format(table_name))

        for key in ["tag", "param", "metric"]:
            table_name = "{}_{}_{}".format(self.table_prefix, DynamodbStore.RUN_TABLE, key)
            response = client.create_table(
                AttributeDefinitions=[
                    {"AttributeName": "run_id", "AttributeType": "S"},
                    {"AttributeName": "key", "AttributeType": "S"},
                ],
                TableName=table_name,
                KeySchema=[
                    {"AttributeName": "run_id", "KeyType": "HASH"},
                    {"AttributeName": "key", "KeyType": "RANGE"},
                ],
                ProvisionedThroughput={"ReadCapacityUnits": rcu, "WriteCapacityUnits": wcu},
            )
            if response["ResponseMetadata"]["HTTPStatusCode"] != 200:
                raise MlflowException("Unable to create table '%s'" % table_name)
            waiter.wait(TableName=table_name)
            print("table {} created".format(table_name))

    def delete_tables(self):
        print("delete tables")
        client = self._get_dynamodb_client()
        for key in [
            DynamodbStore.EXPERIMENT_TABLE,
            DynamodbStore.RUN_TABLE,
            "{}_tag".format(DynamodbStore.RUN_TABLE),
            "{}_param".format(DynamodbStore.RUN_TABLE),
            "{}_metric".format(DynamodbStore.RUN_TABLE),
        ]:
            table_name = "{}_{}".format(self.table_prefix, key)
            response = client.delete_table(TableName=table_name)
            if response["ResponseMetadata"]["HTTPStatusCode"] != 200:
                raise MlflowException("Unable to delete table '%s'" % table_name)

    def _list_experiments(self, view_type=None, name=None):
        dynamodb = self._get_dynamodb_resource()
        table_name = "_".join([self.table_prefix, DynamodbStore.EXPERIMENT_TABLE])
        table = dynamodb.Table(table_name)

        # Filter on active/deleted with optional name
        condition = None
        if self.use_gsi and view_type and view_type != ViewType.ALL:
            if view_type == ViewType.ACTIVE_ONLY:
                condition = Key("lifecycle_stage").eq(LifecycleStage.ACTIVE)
            elif view_type == ViewType.DELETED_ONLY:
                condition = Key("lifecycle_stage").eq(LifecycleStage.DELETED)
            if name:
                condition = And(condition, Key("name").eq(name))
            response = table.query(
                IndexName="LifeCycleStage",
                KeyConditionExpression=condition,
                ReturnConsumedCapacity="TOTAL",
            )
        elif name:
            condition = Key("name").eq(name)
            response = table.scan(FilterExpression=condition, ReturnConsumedCapacity="TOTAL",)
        else:
            response = table.scan(ReturnConsumedCapacity="TOTAL",)

        items = []
        if response["ResponseMetadata"]["HTTPStatusCode"] != 200:
            raise MlflowException("DynamoDB connection error")
        if "Items" in response:
            items += _filter_experiment(response["Items"], view_type, name)

        # Keep fetching results if there are more than the limit
        while "LastEvaluatedKey" in response:
            print("more", response["LastEvaluatedKey"])
            if self.use_gsi and view_type and view_type != ViewType.ALL:
                response = table.query(
                    IndexName="LifeCycleStage",
                    KeyConditionExpression=condition,
                    ReturnConsumedCapacity="TOTAL",
                    ExclusiveStartKey=response["LastEvaluatedKey"],
                )
            elif name:
                response = table.scan(
                    FilterExpression=condition,
                    ReturnConsumedCapacity="TOTAL",
                    ExclusiveStartKey=response["LastEvaluatedKey"],
                )
            else:
                response = table.scan(
                    ReturnConsumedCapacity="TOTAL", ExclusiveStartKey=response["LastEvaluatedKey"],
                )

            if response["ResponseMetadata"]["HTTPStatusCode"] != 200:
                raise MlflowException("DynamoDB connection error")
            if "Items" in response:
                items += _filter_experiment(response["Items"], view_type, name)
        return items

    def list_experiments(self, view_type=ViewType.ACTIVE_ONLY):
        """

        :param view_type: Qualify requested type of experiments.

        :return: a list of Experiment objects stored in store for requested view.
        """
        experiments = self._list_experiments(view_type=view_type)
        return [_dict_to_experiment(e) for e in experiments]

    def _create_experiment_with_id(self, name, experiment_id, artifact_uri):
        dynamodb = self._get_dynamodb_resource()
        table_name = "_".join([self.table_prefix, DynamodbStore.EXPERIMENT_TABLE])
        table = dynamodb.Table(table_name)
        exp = Experiment(
            experiment_id=experiment_id,
            name=name,
            artifact_location=artifact_uri,
            lifecycle_stage=LifecycleStage.ACTIVE,
        )
        response = table.put_item(Item=_entity_to_dict(exp), ReturnConsumedCapacity="TOTAL",)
        if response["ResponseMetadata"]["HTTPStatusCode"] != 200:
            raise MlflowException("DynamoDB connection error")
        return experiment_id

    def create_experiment(self, name, artifact_location=None):
        """
        Create a new experiment.
        If an experiment with the given name already exists, throws exception.

        :param name: Desired name for an experiment
        :param artifact_location: Base location for artifacts in runs. May be None.

        :return: experiment_id (string) for the newly created experiment if successful, else None.
        """

        if name is None or name == "":
            raise MlflowException("Invalid experiment name '%s'" % name, INVALID_PARAMETER_VALUE)

        if self._list_experiments(name=name):
            raise MlflowException("Experiment '%s' already exists." % name, RESOURCE_ALREADY_EXISTS)
        # Get all existing experiments and find the one with largest numerical ID.
        # len(list_all(..)) would not work when experiments are deleted.
        experiments_ids = [int(e.experiment_id) for e in self.list_experiments(ViewType.ALL)]
        experiment_id = str(max(experiments_ids) + 1 if experiments_ids else 0)
        return self._create_experiment_with_id(name, experiment_id, artifact_location)

    def _get_experiment(self, experiment_id):
        dynamodb = self._get_dynamodb_resource()
        table_name = "_".join([self.table_prefix, DynamodbStore.EXPERIMENT_TABLE])
        table = dynamodb.Table(table_name)
        response = table.get_item(
            Key={"experiment_id": experiment_id}, ReturnConsumedCapacity="TOTAL",
        )
        if response["ResponseMetadata"]["HTTPStatusCode"] != 200:
            raise MlflowException("DynamoDB connection error")
        if "Item" in response:
            return response["Item"]

    def get_experiment(self, experiment_id):
        """
        Fetch the experiment by ID from the backend store.

        :param experiment_id: String id for the experiment

        :return: A single :py:class:`mlflow.entities.Experiment` object if it exists,
            otherwise raises an exception.

        """
        exp = self._get_experiment(experiment_id)
        if exp is None:
            raise MlflowException(
                "Could not find experiment with ID %s" % experiment_id, RESOURCE_DOES_NOT_EXIST,
            )
        return _dict_to_experiment(exp)

    def get_experiment_by_name(self, experiment_name):
        """
        Fetch the experiment by name from the backend store.
        This is a base implementation using ``list_experiments``, derived classes may have
        some specialized implementations.

        :param experiment_name: Name of experiment

        :return: A single :py:class:`mlflow.entities.Experiment` object if it exists.
        """
        experiments = self._list_experiments(name=experiment_name)
        if experiments:
            return _dict_to_experiment(experiments[0])
        return None

    def _update_experiment_status(
        self, experiment_id, before_lifecycle_stage, after_lifecycle_stage
    ):

        dynamodb = self._get_dynamodb_resource()
        table_name = "_".join([self.table_prefix, DynamodbStore.EXPERIMENT_TABLE])
        table = dynamodb.Table(table_name)
        response = table.update_item(
            Key={"experiment_id": experiment_id},
            UpdateExpression="SET lifecycle_stage = :a",
            ConditionExpression="lifecycle_stage = :b",
            ExpressionAttributeValues={":b": before_lifecycle_stage, ":a": after_lifecycle_stage},
            ReturnValues="UPDATED_NEW",
            ReturnConsumedCapacity="TOTAL",
        )
        if response["ResponseMetadata"]["HTTPStatusCode"] != 200:
            raise MlflowException("DynamoDB connection error")
        return "Attributes" in response

    def delete_experiment(self, experiment_id):
        """
        Delete the experiment from the backend store. Deleted experiments can be restored until
        permanently deleted.

        :param experiment_id: String id for the experiment
        """

        experiment = self.get_experiment(experiment_id)
        if experiment.lifecycle_stage != LifecycleStage.ACTIVE:
            raise MlflowException(
                "Could not find experiment with ID %s" % experiment_id, RESOURCE_DOES_NOT_EXIST,
            )
        return self._update_experiment_status(
            experiment_id, LifecycleStage.ACTIVE, LifecycleStage.DELETED
        )

    def restore_experiment(self, experiment_id):
        """
        Restore deleted experiment unless it is permanently deleted.

        :param experiment_id: String id for the experiment
        """

        experiment = self.get_experiment(experiment_id)
        if experiment.lifecycle_stage != LifecycleStage.DELETED:
            raise MlflowException(
                "Could not find deleted experiment with ID %s" % experiment_id,
                RESOURCE_DOES_NOT_EXIST,
            )
        return self._update_experiment_status(
            experiment_id, LifecycleStage.DELETED, LifecycleStage.ACTIVE
        )

    def _rename_experiment(self, experiment_id, new_name):
        dynamodb = self._get_dynamodb_resource()
        table_name = "_".join([self.table_prefix, DynamodbStore.EXPERIMENT_TABLE])
        table = dynamodb.Table(table_name)
        response = table.update_item(
            Key={"experiment_id": experiment_id},
            UpdateExpression="SET #name = :n",
            ExpressionAttributeNames={"#name": "name"},
            ExpressionAttributeValues={":n": new_name},
            ReturnValues="ALL_NEW",
            ReturnConsumedCapacity="TOTAL",
        )

        if response["ResponseMetadata"]["HTTPStatusCode"] != 200:
            raise MlflowException("DynamoDB connection error")
        if "Attributes" in response:
            return response["Attributes"]

    def rename_experiment(self, experiment_id, new_name):
        """
        Update an experiment's name. The new name must be unique.

        :param experiment_id: String id for the experiment
        """

        experiment = self.get_experiment(experiment_id)
        if experiment.lifecycle_stage != LifecycleStage.ACTIVE:
            raise Exception(
                "Cannot rename experiment in non-active lifecycle stage."
                " Current stage: %s" % experiment.lifecycle_stage
            )
        return _dict_to_experiment(self._rename_experiment(experiment_id, new_name))

    def _get_run_info(self, run_id):
        dynamodb = self._get_dynamodb_resource()
        table_name = "_".join([self.table_prefix, DynamodbStore.RUN_TABLE])
        table = dynamodb.Table(table_name)
        response = table.get_item(Key={"run_id": run_id}, ReturnConsumedCapacity="TOTAL",)
        if response["ResponseMetadata"]["HTTPStatusCode"] != 200:
            raise MlflowException("DynamoDB connection error")
        if "Item" in response:
            return _dict_to_run_info(response["Item"])
        raise MlflowException("Run '%s' not found" % run_id, RESOURCE_DOES_NOT_EXIST)

    def _update_run_status(self, run_id, before_lifecycle_stage, after_lifecycle_stage):
        dynamodb = self._get_dynamodb_resource()
        table_name = "_".join([self.table_prefix, DynamodbStore.RUN_TABLE])
        table = dynamodb.Table(table_name)
        response = table.update_item(
            Key={"run_id": run_id},
            UpdateExpression="SET lifecycle_stage = :a",
            ConditionExpression="lifecycle_stage = :b",
            ExpressionAttributeValues={":b": before_lifecycle_stage, ":a": after_lifecycle_stage},
            ReturnValues="UPDATED_NEW",
            ReturnConsumedCapacity="TOTAL",
        )
        if response["ResponseMetadata"]["HTTPStatusCode"] != 200:
            raise MlflowException("DynamoDB connection error")
        return "Attributes" in response

    def delete_run(self, run_id):
        """
        Delete a run.

        :param run_id
        """
        _validate_run_id(run_id)
        run_info = self._get_run_info(run_id)
        check_run_is_active(run_info)
        return self._update_run_status(run_id, LifecycleStage.ACTIVE, LifecycleStage.DELETED)

    def restore_run(self, run_id):
        """
        Restore a run.

        :param run_id
        """
        _validate_run_id(run_id)
        run_info = self._get_run_info(run_id)
        check_run_is_deleted(run_info)
        return self._update_run_status(run_id, LifecycleStage.DELETED, LifecycleStage.ACTIVE)

    def _update_run_info(self, run_id, run_status, end_time):
        dynamodb = self._get_dynamodb_resource()
        table_name = "_".join([self.table_prefix, DynamodbStore.RUN_TABLE])
        table = dynamodb.Table(table_name)
        response = table.update_item(
            Key={"run_id": run_id},
            ConditionExpression="lifecycle_stage = :l",
            UpdateExpression="SET #status = :run_status, #end_time = :end_time",
            ExpressionAttributeNames={"#status": "status", "#end_time": "end_time"},
            ExpressionAttributeValues={
                ":l": LifecycleStage.ACTIVE,
                ":run_status": run_status,
                ":end_time": end_time,
            },
            ReturnValues="ALL_NEW",
            ReturnConsumedCapacity="TOTAL",
        )
        if response["ResponseMetadata"]["HTTPStatusCode"] != 200:
            raise MlflowException("DynamoDB connection error")
        if "Attributes" in response:
            return _dict_to_run_info(response["Attributes"])

    def update_run_info(self, run_id, run_status, end_time):
        """
        Update the metadata of the specified run.

        :return: :py:class:`mlflow.entities.RunInfo` describing the updated run.
        """
        _validate_run_id(run_id)
        run_info = self._get_run_info(run_id)
        check_run_is_active(run_info)
        return self._update_run_info(run_id, run_status, end_time)

    def _create_run_info(self, run_info_dict):
        dynamodb = self._get_dynamodb_resource()
        table_name = "_".join([self.table_prefix, DynamodbStore.RUN_TABLE])
        table = dynamodb.Table(table_name)
        response = table.put_item(Item=run_info_dict, ReturnConsumedCapacity="TOTAL",)
        if response["ResponseMetadata"]["HTTPStatusCode"] != 200:
            raise MlflowException("DynamoDB connection error")
        return True

    def create_run(self, experiment_id, user_id, start_time, tags):
        """
        Create a run under the specified experiment ID, setting the run's status to "RUNNING"
        and the start time to the current time.

        :param experiment_id: String id of the experiment for this run
        :param user_id: ID of the user launching this run

        :return: The created Run object
        """
        experiment = self.get_experiment(experiment_id)
        if experiment is None:
            raise MlflowException(
                "Could not create run under experiment with ID %s - no such experiment "
                "exists." % experiment_id,
                RESOURCE_DOES_NOT_EXIST,
            )
        if experiment.lifecycle_stage != LifecycleStage.ACTIVE:
            raise MlflowException(
                "Could not create run under non-active experiment with ID " "%s." % experiment_id,
                INVALID_STATE,
            )
        run_id = uuid.uuid4().hex
        artifact_uri = os.path.join(experiment.artifact_location, run_id, "artifacts")
        run_info = RunInfo(
            run_uuid=run_id,
            run_id=run_id,
            experiment_id=experiment_id,
            artifact_uri=artifact_uri,
            user_id=user_id,
            status=RunStatus.to_string(RunStatus.RUNNING),
            start_time=start_time,
            end_time=None,
            lifecycle_stage=LifecycleStage.ACTIVE,
        )
        if self._create_run_info(_entity_to_dict(run_info)):
            for tag in tags:
                self.set_tag(run_id, tag)
            return Run(run_info=run_info, run_data=None)

    def get_run(self, run_id):
        """
        Fetch the run from backend store. The resulting :py:class:`Run <mlflow.entities.Run>`
        contains a collection of run metadata - :py:class:`RunInfo <mlflow.entities.RunInfo>`,
        as well as a collection of run parameters, tags, and metrics -
        :py:class`RunData <mlflow.entities.RunData>`. In the case where multiple metrics with the
        same key are logged for the run, the :py:class:`RunData <mlflow.entities.RunData>` contains
        the value at the latest timestamp for each metric. If there are multiple values with the
        latest timestamp for a given metric, the maximum of these values is returned.

        :param run_id: Unique identifier for the run.

        :return: A single :py:class:`mlflow.entities.Run` object, if the run exists. Otherwise,
                 raises an exception.
        """

        _validate_run_id(run_id)
        run_info = self._get_run_info(run_id)
        metrics = self.get_all_metrics(run_id)
        params = self.get_all_params(run_id)
        tags = self.get_all_tags(run_id)
        return Run(run_info, RunData(metrics, params, tags))

    def _get_run_metrics(self, run_id, metric_key=None):
        dynamodb = self._get_dynamodb_resource()
        table_name = "_".join([self.table_prefix, DynamodbStore.METRICS_TABLE])
        table = dynamodb.Table(table_name)
        condition = Key("run_id").eq(run_id)
        if metric_key:
            condition = And(condition, Key("key").eq(metric_key))
        # If we don't support projections
        if self.use_projections:
            response = table.query(
                ProjectionExpression="#key, #metrics[0].#value, #metrics[0].#timestamp",
                ExpressionAttributeNames={
                    "#key": "key",
                    "#metrics": "metrics",
                    "#value": "value",
                    "#timestamp": "timestamp",
                },
                ConsistentRead=True,
                KeyConditionExpression=condition,
                ReturnConsumedCapacity="TOTAL",
            )
        else:
            response = table.query(
                ConsistentRead=True,
                KeyConditionExpression=condition,
                ReturnConsumedCapacity="TOTAL",
            )
        if response["ResponseMetadata"]["HTTPStatusCode"] != 200:
            raise MlflowException("DynamoDB connection error")
        if "Items" in response:
            return response["Items"]
        return []

    def get_metric(self, run_id, metric_key):
        _validate_run_id(run_id)
        _validate_metric_name(metric_key)
        metrics = self._get_run_metrics(run_id, metric_key)
        if not metrics:
            raise MlflowException(
                "Metric '%s' not found under run '%s'" % (metric_key, run_id),
                RESOURCE_DOES_NOT_EXIST,
            )
        return _list_to_run_metric(metrics)[0]

    def get_all_metrics(self, run_id):
        _validate_run_id(run_id)
        return _list_to_run_metric(self._get_run_metrics(run_id))

    def get_metric_history(self, run_id, metric_key):
        _validate_run_id(run_id)
        _validate_metric_name(metric_key)
        dynamodb = self._get_dynamodb_resource()
        table_name = "_".join([self.table_prefix, DynamodbStore.METRICS_TABLE])
        table = dynamodb.Table(table_name)
        response = table.get_item(
            Key={"run_id": run_id, "key": metric_key}, ReturnConsumedCapacity="TOTAL",
        )
        if response["ResponseMetadata"]["HTTPStatusCode"] != 200:
            raise MlflowException("DynamoDB connection error")
        if "Item" in response:
            return _dict_to_run_metric_history(response["Item"])

    def _get_run_params(self, run_id, param_name=None):
        dynamodb = self._get_dynamodb_resource()
        table_name = "_".join([self.table_prefix, DynamodbStore.PARAMS_TABLE])
        table = dynamodb.Table(table_name)
        condition = Key("run_id").eq(run_id)
        if param_name:
            condition = And(condition, Key("key").eq(param_name))
        response = table.query(
            ConsistentRead=True, KeyConditionExpression=condition, ReturnConsumedCapacity="TOTAL",
        )
        if response["ResponseMetadata"]["HTTPStatusCode"] != 200:
            raise MlflowException("DynamoDB connection error")
        if "Items" in response:
            return response["Items"]
        return []

    def get_param(self, run_id, param_name):
        _validate_run_id(run_id)
        _validate_param_name(param_name)
        params = self._get_run_params(run_id, param_name)
        if not params:
            raise MlflowException(
                "Param '%s' not found under run '%s'" % (param_name, run_id),
                RESOURCE_DOES_NOT_EXIST,
            )
        return _list_to_run_param(params)[0]

    def get_all_params(self, run_id):
        _validate_run_id(run_id)
        return _list_to_run_param(self._get_run_params(run_id))

    def get_all_tags(self, run_id):
        _validate_run_id(run_id)
        dynamodb = self._get_dynamodb_resource()
        table_name = "_".join([self.table_prefix, DynamodbStore.TAGS_TABLE])
        table = dynamodb.Table(table_name)
        response = table.query(
            ConsistentRead=True,
            KeyConditionExpression=Key("run_id").eq(run_id),
            ReturnConsumedCapacity="TOTAL",
        )
        if response["ResponseMetadata"]["HTTPStatusCode"] != 200:
            raise MlflowException("DynamoDB connection error")
        if "Items" in response:
            return _list_to_run_tag(response["Items"])
        return []

    def _list_runs_ids(self, experiment_id, view_type=None):
        dynamodb = self._get_dynamodb_resource()
        table_name = "_".join([self.table_prefix, DynamodbStore.RUN_TABLE])
        table = dynamodb.Table(table_name)

        # Filter on active/deleted with optional experiment_id
        condition = None
        if self.use_gsi and view_type and view_type != ViewType.ALL:
            if view_type == ViewType.ACTIVE_ONLY:
                condition = Key("lifecycle_stage").eq(LifecycleStage.ACTIVE)
            elif view_type == ViewType.DELETED_ONLY:
                condition = Key("lifecycle_stage").eq(LifecycleStage.DELETED)
            if experiment_id:
                condition = And(condition, Key("experiment_id").eq(experiment_id))
            response = table.query(
                IndexName="LifeCycleStage",
                KeyConditionExpression=condition,
                ProjectionExpression="run_id, experiment_id, lifecycle_stage",
                ReturnConsumedCapacity="TOTAL",
            )
        elif experiment_id:
            condition = Key("experiment_id").eq(experiment_id)
            response = table.scan(FilterExpression=condition, ReturnConsumedCapacity="TOTAL",)
        else:
            response = table.scan(ReturnConsumedCapacity="TOTAL",)

        if response["ResponseMetadata"]["HTTPStatusCode"] != 200:
            raise MlflowException("DynamoDB connection error")
        if "Items" in response:
            return _filter_run(response["Items"], view_type, experiment_id)
        return []

    def _get_run_list(self, runs):
        keys = [{"run_id": r["run_id"]} for r in runs]
        if len(keys) == 0:
            return []
        dynamodb = self._get_dynamodb_resource()
        table_name = "_".join([self.table_prefix, DynamodbStore.RUN_TABLE])
        response = dynamodb.batch_get_item(
            RequestItems={table_name: {"Keys": keys}}, ReturnConsumedCapacity="TOTAL"
        )
        if response["ResponseMetadata"]["HTTPStatusCode"] != 200:
            raise MlflowException("DynamoDB connection error")
        if "Responses" in response:
            return response["Responses"][table_name]
        return []

    def _search_runs(
        self, experiment_ids, filter_string, run_view_type, max_results, order_by, page_token,
    ):
        runs = []
        for experiment_id in experiment_ids:
            run_ids = self._list_runs_ids(experiment_id, run_view_type)
            run_infos = [_dict_to_run_info(r) for r in self._get_run_list(run_ids)]
            for run_info in run_infos:
                # Load the metrics, params and tags for the run
                run_id = run_info.run_id
                metrics = self.get_all_metrics(run_id)
                params = self.get_all_params(run_id)
                tags = self.get_all_tags(run_id)
                run = Run(run_info, RunData(metrics, params, tags))
                runs.append(run)

        filtered = SearchUtils.filter(runs, filter_string)
        sorted_runs = SearchUtils.sort(filtered, order_by)
        runs, next_page_token = SearchUtils.paginate(sorted_runs, page_token, max_results)
        return runs, next_page_token

    def list_run_infos(self, experiment_id, run_view_type):
        run_ids = self._list_runs_ids(experiment_id, run_view_type)
        return [_dict_to_run_info(r) for r in self._get_run_list(run_ids)]

    def log_metric(self, run_id, metric):
        """
        Log a metric for the specified run

        :param run_id: String id for the run
        :param metric: :py:class:`mlflow.entities.Metric` instance to log
        """
        _validate_run_id(run_id)
        _validate_metric_name(metric.key)
        run_info = self._get_run_info(run_id)
        check_run_is_active(run_info)
        dynamodb = self._get_dynamodb_resource()
        table_name = "_".join([self.table_prefix, DynamodbStore.METRICS_TABLE])
        table = dynamodb.Table(table_name)
        # Append metrics to head of list, so the first element is most recent
        response = table.update_item(
            Key={"run_id": run_id, "key": metric.key},
            UpdateExpression="SET #m = list_append(:m, if_not_exists(#m, :e))",
            ExpressionAttributeNames={"#m": "metrics"},
            ExpressionAttributeValues={
                ":e": [],
                ":m": [{"value": Decimal(str(metric.value)), "timestamp": metric.timestamp}],
            },
            ReturnValues="NONE",
            ReturnConsumedCapacity="TOTAL",
        )
        if response["ResponseMetadata"]["HTTPStatusCode"] != 200:
            raise MlflowException("DynamoDB connection error")
        return True

    def log_param(self, run_id, param):
        """
        Log a param for the specified run

        :param run_id: String id for the run
        :param param: :py:class:`mlflow.entities.Param` instance to log
        """
        _validate_run_id(run_id)
        _validate_param_name(param.key)
        run_info = self._get_run_info(run_id)
        check_run_is_active(run_info)
        dynamodb = self._get_dynamodb_resource()
        table_name = "_".join([self.table_prefix, DynamodbStore.PARAMS_TABLE])
        table = dynamodb.Table(table_name)
        response = table.put_item(
            Item={"run_id": run_id, "key": param.key, "value": param.value},
            ReturnConsumedCapacity="TOTAL",
        )
        if response["ResponseMetadata"]["HTTPStatusCode"] != 200:
            raise MlflowException("DynamoDB connection error")
        return True

    def set_tag(self, run_id, tag):
        """
        Set a tag for the specified run

        :param run_id: String id for the run
        :param tag: :py:class:`mlflow.entities.RunTag` instance to set
        """
        _validate_run_id(run_id)
        _validate_tag_name(tag.key)
        run_info = self._get_run_info(run_id)
        check_run_is_active(run_info)
        dynamodb = self._get_dynamodb_resource()
        table_name = "_".join([self.table_prefix, DynamodbStore.TAGS_TABLE])
        table = dynamodb.Table(table_name)
        response = table.put_item(
            Item={"run_id": run_id, "key": tag.key, "value": tag.value},
            ReturnConsumedCapacity="TOTAL",
        )
        if response["ResponseMetadata"]["HTTPStatusCode"] != 200:
            raise MlflowException("DynamoDB connection error")
        return True

    def log_batch(self, run_id, metrics, params, tags):
        _validate_run_id(run_id)
        _validate_batch_log_data(metrics, params, tags)
        _validate_batch_log_limits(metrics, params, tags)
        try:
            for param in params:
                self.log_param(run_id, param)
            for metric in metrics:
                self.log_metric(run_id, metric)
            for tag in tags:
                self.set_tag(run_id, tag)
        except MlflowException as e:
            raise e
        except Exception as e:
            raise MlflowException(e, INTERNAL_ERROR)
