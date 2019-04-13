import os
import uuid

import boto3
from boto3.dynamodb.conditions import Key, And
from decimal import Decimal

from mlflow.entities.lifecycle_stage import LifecycleStage
from mlflow.entities import Experiment, Metric, Param, Run, RunData, RunInfo, RunStatus, RunTag, \
    ViewType
from mlflow.entities.run_info import check_run_is_active, check_run_is_deleted
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE, RESOURCE_ALREADY_EXISTS, \
    INVALID_STATE, RESOURCE_DOES_NOT_EXIST, INTERNAL_ERROR
from mlflow.store.abstract_store import AbstractStore
from mlflow.utils.validation import _validate_metric_name, _validate_param_name, _validate_run_id, \
    _validate_tag_name, _validate_batch_log_limits, _validate_batch_log_data

from mlflow.utils.env import get_env
from mlflow.utils.mlflow_tags import MLFLOW_RUN_NAME, MLFLOW_PARENT_RUN_ID

_DYNAMODB_ENDPOINT_URL_VAR = "MLFLOW_DYNAMODB_ENDPOINT_URL"
_DYNAMODB_TABLE_PREFIX_VAR = "MLFLOW_DYNAMODB_TABLE_PREFIX"


def _default_endpoint_url():
    return get_env(_DYNAMODB_ENDPOINT_URL_VAR)


def _default_table_prefix():
    return get_env(_DYNAMODB_TABLE_PREFIX_VAR) or "mlflow"


def _dict_to_experiment(d):
    return Experiment(
        experiment_id=int(d['experiment_id']),
        name=d['name'],
        artifact_location=d.get('artifact_location') or '',
        lifecycle_stage=d.get('lifecycle_stage', 'active')
    )


def _dict_to_run_info(d):
    return RunInfo(
        run_uuid=d['run_uuid'],
        experiment_id=int(d['experiment_id']),
        name=d.get('name') or '',
        source_type=int(d.get('source_type') or 0),
        source_name=d.get('source_name') or '',
        entry_point_name=d.get('entry_point_name') or '',
        user_id=d.get('user_id') or '',
        status=int(d.get('status') or 0),
        start_time=int(d.get('start_time') or 0),
        end_time=int(d.get('end_time') or 0),
        source_version=d.get('source_version') or '',
        lifecycle_stage=d.get('lifecycle_stage', 'active'),
        artifact_uri=d.get('artifact_uri') or '',
    )


def _list_to_run_tag(l):
    return [RunTag(
        key=d['key'],
        value=d['value']
    ) for d in l]


def _list_to_run_param(l):
    return [Param(
        key=d['key'],
        value=d['value']
    ) for d in l]


# Return the first element in the list, which is the most recent if more than one.
def _list_to_run_metric(l):
    return [Metric(key=rm['key'],
            value=float(rm['metrics'][0]['value']),
            timestamp=int(rm['metrics'][0]['timestamp'])) for rm in l]


# Return the metrics with most recent at the end of the list
def _dict_to_run_metric_history(rm):
    return [Metric(key=rm['key'], value=float(m['value']), timestamp=int(m['timestamp']))
            for m in rm['metrics'][::-1]]


def _entity_to_dict(obj):
    return {k: None if v == '' else v for k, v in dict(obj).items()}


def _filter_view_type(d, view_type=None):
    return view_type is None or view_type == ViewType.ALL or \
        (view_type == ViewType.ACTIVE_ONLY and
            d.get('lifecycle_stage') == LifecycleStage.ACTIVE) or \
        (view_type == ViewType.DELETED_ONLY and
            d.get('lifecycle_stage') == LifecycleStage.DELETED)


def _filter_experiment(experiments, view_type=None, name=None):
    return [e for e in experiments if _filter_view_type(e, view_type)
            and (name is None or e['name'] == name)]


def _filter_run(runs, view_type=None, experiment_id=None):
    return [r for r in runs if _filter_view_type(r, view_type)
            and (experiment_id is None or r['experiment_id'] == experiment_id)]


class DynamodbStore(AbstractStore):
    EXPERIMENT_TABLE = "experiment"
    RUN_TABLE = "run"
    METRICS_TABLE = "run_metric"
    PARAMS_TABLE = "run_param"
    TAGS_TABLE = "run_tag"

    def __init__(self, table_prefix=None, endpoint_url=None, region_name=None,
                 use_gsi=True, use_projections=True):
        """
        Create a new DynamodbStore for storing experiments and runs.

        :param table_prefix: DynamoDb table prefix defaults to 'mlflow'
        :param endpoint_url: Optional endpoint url for testing Dynamodb Local
        :param region_name: Optional Name of the AWS region for Dynamodb
        :param use_gsi: Flag to query Global Secondary Indices, defaults to True.
        :param use_projections: Flag to use projections in queries, defaults to True.
        """
        super(DynamodbStore, self).__init__()
        self.table_prefix = table_prefix or _default_table_prefix()
        self.endpoint_url = endpoint_url or _default_endpoint_url()
        self.region_name = region_name
        self.use_gsi = use_gsi
        self.use_projections = use_projections

    def _get_dynamodb_client(self):
        return boto3.client('dynamodb', endpoint_url=self.endpoint_url,
                            region_name=self.region_name)

    def _get_dynamodb_resource(self):
        return boto3.resource('dynamodb', endpoint_url=self.endpoint_url,
                              region_name=self.region_name)

    def create_tables(self, rcu=1, wcu=1):
        # create a mock dynamodb client, and create tables
        client = self._get_dynamodb_client()
        table_name = '{}_experiment'.format(self.table_prefix)
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
            TableName=table_name,
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
                        'ProjectionType': 'ALL',
                    },
                    'ProvisionedThroughput': {
                        'ReadCapacityUnits': rcu,
                        'WriteCapacityUnits': wcu
                    }
                },
            ],
            ProvisionedThroughput={
                'ReadCapacityUnits': rcu,
                'WriteCapacityUnits': wcu
            },
        )
        if response['ResponseMetadata']['HTTPStatusCode'] != 200:
            raise MlflowException("Unable to create table '%s'" % table_name)

        table_name = '{}_run'.format(self.table_prefix)
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
            TableName=table_name,
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
                        'ReadCapacityUnits': rcu,
                        'WriteCapacityUnits': wcu
                    }
                },
            ],
            ProvisionedThroughput={
                'ReadCapacityUnits': rcu,
                'WriteCapacityUnits': wcu
            },
        )
        if response['ResponseMetadata']['HTTPStatusCode'] != 200:
            raise MlflowException("Unable to create table '%s'" % table_name)

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
                    'ReadCapacityUnits': rcu,
                    'WriteCapacityUnits': wcu
                },
            )
            if response['ResponseMetadata']['HTTPStatusCode'] != 200:
                raise MlflowException("Unable to create table '%s'" % table_name)

    def delete_tables(self):
        client = self._get_dynamodb_client()
        for key in ['experiment', 'run', 'run_tag', 'run_param', 'run_metric']:
            table_name = '{}_{}'.format(self.table_prefix, key)
            response = client.delete_table(TableName=table_name)
            if response['ResponseMetadata']['HTTPStatusCode'] != 200:
                raise MlflowException("Unable to delete table '%s'" % table_name)

    def _list_experiments(self, view_type=None, name=None):
        dynamodb = self._get_dynamodb_resource()
        table_name = '_'.join([self.table_prefix, DynamodbStore.EXPERIMENT_TABLE])
        table = dynamodb.Table(table_name)

        # Filter on active/deleted with optional name
        condition = None
        if self.use_gsi and view_type and view_type != ViewType.ALL:
            if view_type == ViewType.ACTIVE_ONLY:
                condition = Key('lifecycle_stage').eq(LifecycleStage.ACTIVE)
            elif view_type == ViewType.DELETED_ONLY:
                condition = Key('lifecycle_stage').eq(LifecycleStage.DELETED)
            if name:
                condition = And(condition, Key('name').eq(name))
            response = table.query(
                IndexName='LifeCycleStage',
                KeyConditionExpression=condition,
                ReturnConsumedCapacity='TOTAL',
            )
        elif name:
            condition = Key('name').eq(name)
            response = table.scan(
                FilterExpression=condition,
                ReturnConsumedCapacity='TOTAL',
            )
        else:
            response = table.scan(
                ReturnConsumedCapacity='TOTAL',
            )

        items = []
        if response['ResponseMetadata']['HTTPStatusCode'] != 200:
            raise MlflowException("DynamoDB connection error")
        if 'Items' in response:
            items += _filter_experiment(response['Items'], view_type, name)

        # Keey fetching results if there are more than the limit
        while 'LastEvaluatedKey' in response:
            print('more', response['LastEvaluatedKey'])
            if self.use_gsi and view_type and view_type != ViewType.ALL:
                response = table.query(
                    IndexName='LifeCycleStage',
                    KeyConditionExpression=condition,
                    ReturnConsumedCapacity='TOTAL',
                    ExclusiveStartKey=response['LastEvaluatedKey']
                )
            elif name:
                response = table.scan(
                    FilterExpression=condition,
                    ReturnConsumedCapacity='TOTAL',
                    ExclusiveStartKey=response['LastEvaluatedKey']
                )
            else:
                response = table.scan(
                    ReturnConsumedCapacity='TOTAL',
                    ExclusiveStartKey=response['LastEvaluatedKey']
                )

            if response['ResponseMetadata']['HTTPStatusCode'] != 200:
                raise MlflowException("DynamoDB connection error")
            if 'Items' in response:
                items += _filter_experiment(response['Items'], view_type, name)
        return items

    def list_experiments(self, view_type=ViewType.ACTIVE_ONLY):
        exps = self._list_experiments(view_type=view_type)
        return [_dict_to_experiment(e) for e in exps]

    def _create_experiment_with_id(self, name, experiment_id, artifact_uri):
        dynamodb = self._get_dynamodb_resource()
        table_name = '_'.join([self.table_prefix, DynamodbStore.EXPERIMENT_TABLE])
        table = dynamodb.Table(table_name)
        exp = Experiment(
            experiment_id=experiment_id,
            name=name,
            artifact_location=artifact_uri,
            lifecycle_stage=LifecycleStage.ACTIVE
        )
        response = table.put_item(
            Item=_entity_to_dict(exp),
            ReturnConsumedCapacity='TOTAL',
        )
        if response['ResponseMetadata']['HTTPStatusCode'] != 200:
            raise MlflowException("DynamoDB connection error")
        return experiment_id

    def create_experiment(self, name, artifact_location=None):
        if name is None or name == "":
            raise MlflowException("Invalid experiment name '%s'" % name,
                                  INVALID_PARAMETER_VALUE)

        if self._list_experiments(name=name):
            raise MlflowException("Experiment '%s' already exists." % name,
                                  RESOURCE_ALREADY_EXISTS)
        # Get all existing experiments and find the one with largest ID.
        # len(list_all(..)) would not work when experiments are deleted.
        experiments_ids = [e.experiment_id for e in self.list_experiments(ViewType.ALL)]
        experiment_id = max(experiments_ids) + 1 if experiments_ids else 0
        return self._create_experiment_with_id(name, experiment_id, artifact_location)

    def _get_experiment(self, experiment_id):
        dynamodb = self._get_dynamodb_resource()
        table_name = '_'.join([self.table_prefix, DynamodbStore.EXPERIMENT_TABLE])
        table = dynamodb.Table(table_name)
        response = table.get_item(
            Key={
                'experiment_id': experiment_id,
            },
            ReturnConsumedCapacity='TOTAL',
        )
        if response['ResponseMetadata']['HTTPStatusCode'] != 200:
            raise MlflowException("DynamoDB connection error")
        if 'Item' in response:
            return response['Item']

    def get_experiment(self, experiment_id):
        """
        Fetches the experiment. This will search for active as well as deleted experiments.

        :param experiment_id: Integer id for the experiment
        :return: A single Experiment object if it exists, otherwise raises an Exception.
        """
        exp = self._get_experiment(experiment_id)
        if exp is None:
            raise MlflowException("Could not find experiment with ID %s" % experiment_id,
                                  RESOURCE_DOES_NOT_EXIST)
        return _dict_to_experiment(exp)

    def get_experiment_by_name(self, name):
        exps = self._list_experiments(name=name)
        if exps:
            return _dict_to_experiment(exps[0])
        return None

    def _update_experiment_status(self, experiment_id,
                                  before_lifecycle_stage, after_lifecycle_stage):

        dynamodb = self._get_dynamodb_resource()
        table_name = '_'.join([self.table_prefix, DynamodbStore.EXPERIMENT_TABLE])
        table = dynamodb.Table(table_name)
        response = table.update_item(
            Key={
                'experiment_id': experiment_id,
            },
            UpdateExpression="SET lifecycle_stage = :a",
            ConditionExpression="lifecycle_stage = :b",
            ExpressionAttributeValues={
                ':b': before_lifecycle_stage,
                ':a': after_lifecycle_stage,
            },
            ReturnValues="UPDATED_NEW",
            ReturnConsumedCapacity='TOTAL',
        )
        if response['ResponseMetadata']['HTTPStatusCode'] != 200:
            raise MlflowException("DynamoDB connection error")
        return 'Attributes' in response

    def delete_experiment(self, experiment_id):
        experiment = self.get_experiment(experiment_id)
        if experiment.lifecycle_stage != LifecycleStage.ACTIVE:
            raise MlflowException("Could not find experiment with ID %s" % experiment_id,
                                  RESOURCE_DOES_NOT_EXIST)
        return self._update_experiment_status(experiment_id,
                                              LifecycleStage.ACTIVE,
                                              LifecycleStage.DELETED)

    def restore_experiment(self, experiment_id):
        experiment = self.get_experiment(experiment_id)
        if experiment.lifecycle_stage != LifecycleStage.DELETED:
            raise MlflowException("Could not find deleted experiment with ID %s" % experiment_id,
                                  RESOURCE_DOES_NOT_EXIST)
        return self._update_experiment_status(experiment_id,
                                              LifecycleStage.DELETED,
                                              LifecycleStage.ACTIVE)

    def _rename_experiment(self, experiment_id, new_name):
        dynamodb = self._get_dynamodb_resource()
        table_name = '_'.join([self.table_prefix, DynamodbStore.EXPERIMENT_TABLE])
        table = dynamodb.Table(table_name)
        response = table.update_item(
            Key={
                'experiment_id': experiment_id,
            },
            UpdateExpression="SET #name = :n",
            ExpressionAttributeNames={'#name': 'name'},
            ExpressionAttributeValues={
                ':n': new_name,
            },
            ReturnValues="ALL_NEW",
            ReturnConsumedCapacity='TOTAL',
        )

        if response['ResponseMetadata']['HTTPStatusCode'] != 200:
            raise MlflowException("DynamoDB connection error")
        if 'Attributes' in response:
            return response['Attributes']

    def rename_experiment(self, experiment_id, new_name):
        experiment = self.get_experiment(experiment_id)
        if experiment.lifecycle_stage != LifecycleStage.ACTIVE:
            raise Exception("Cannot rename experiment in non-active lifecycle stage."
                            " Current stage: %s" % experiment.lifecycle_stage)
        return _dict_to_experiment(self._rename_experiment(experiment_id, new_name))

    def _get_run_info(self, run_uuid):
        dynamodb = self._get_dynamodb_resource()
        table_name = '_'.join([self.table_prefix, DynamodbStore.RUN_TABLE])
        table = dynamodb.Table(table_name)
        response = table.get_item(
            Key={
                'run_uuid': run_uuid,
            },
            ReturnConsumedCapacity='TOTAL',
        )
        if response['ResponseMetadata']['HTTPStatusCode'] != 200:
            raise MlflowException("DynamoDB connection error")
        if 'Item' in response:
            return _dict_to_run_info(response['Item'])
        raise MlflowException("Run '%s' not found" % run_uuid,
                              RESOURCE_DOES_NOT_EXIST)

    def _update_run_status(self, run_uuid, before_lifecycle_stage, after_lifecycle_stage):
        dynamodb = self._get_dynamodb_resource()
        table_name = '_'.join([self.table_prefix, DynamodbStore.RUN_TABLE])
        table = dynamodb.Table(table_name)
        response = table.update_item(
            Key={
                'run_uuid': run_uuid,
            },
            UpdateExpression="SET lifecycle_stage = :a",
            ConditionExpression="lifecycle_stage = :b",
            ExpressionAttributeValues={
                ':b': before_lifecycle_stage,
                ':a': after_lifecycle_stage,
            },
            ReturnValues="UPDATED_NEW",
            ReturnConsumedCapacity='TOTAL',
        )
        if response['ResponseMetadata']['HTTPStatusCode'] != 200:
            raise MlflowException("DynamoDB connection error")
        return 'Attributes' in response

    def delete_run(self, run_id):
        _validate_run_id(run_id)
        run_info = self._get_run_info(run_id)
        check_run_is_active(run_info)
        return self._update_run_status(run_id,
                                       LifecycleStage.ACTIVE,
                                       LifecycleStage.DELETED)

    def restore_run(self, run_id):
        _validate_run_id(run_id)
        run_info = self._get_run_info(run_id)
        check_run_is_deleted(run_info)
        return self._update_run_status(run_id,
                                       LifecycleStage.DELETED,
                                       LifecycleStage.ACTIVE)

    def _update_run_info(self, run_uuid, run_status, end_time):
        dynamodb = self._get_dynamodb_resource()
        table_name = '_'.join([self.table_prefix, DynamodbStore.RUN_TABLE])
        table = dynamodb.Table(table_name)
        response = table.update_item(
            Key={
                'run_uuid': run_uuid,
            },
            ConditionExpression="lifecycle_stage = :l",
            UpdateExpression="SET #status = :run_status, #end_time = :end_time",
            ExpressionAttributeNames={
                '#status': 'status',
                '#end_time': 'end_time',
            },
            ExpressionAttributeValues={
                ':l': LifecycleStage.ACTIVE,
                ':run_status': run_status,
                ':end_time': end_time,
            },
            ReturnValues="ALL_NEW",
            ReturnConsumedCapacity='TOTAL',
        )
        if response['ResponseMetadata']['HTTPStatusCode'] != 200:
            raise MlflowException("DynamoDB connection error")
        if 'Attributes' in response:
            return _dict_to_run_info(response['Attributes'])

    def update_run_info(self, run_uuid, run_status, end_time):
        _validate_run_id(run_uuid)
        run_info = self._get_run_info(run_uuid)
        check_run_is_active(run_info)
        return self._update_run_info(run_uuid, run_status, end_time)

    def _create_run_info(self, run_info_dict):
        dynamodb = self._get_dynamodb_resource()
        table_name = '_'.join([self.table_prefix, DynamodbStore.RUN_TABLE])
        table = dynamodb.Table(table_name)
        response = table.put_item(
            Item=run_info_dict,
            ReturnConsumedCapacity='TOTAL',
        )
        if response['ResponseMetadata']['HTTPStatusCode'] != 200:
            raise MlflowException("DynamoDB connection error")
        return True

    def create_run(self, experiment_id, user_id, run_name, source_type,
                   source_name, entry_point_name, start_time, source_version, tags, parent_run_id):
        """
        Creates a run with the specified attributes.
        """
        experiment = self.get_experiment(experiment_id)
        if experiment is None:
            raise MlflowException(
                    "Could not create run under experiment with ID %s - no such experiment "
                    "exists." % experiment_id,
                    RESOURCE_DOES_NOT_EXIST)
        if experiment.lifecycle_stage != LifecycleStage.ACTIVE:
            raise MlflowException(
                    "Could not create run under non-active experiment with ID "
                    "%s." % experiment_id,
                    INVALID_STATE)
        run_uuid = uuid.uuid4().hex
        artifact_uri = os.path.join(experiment.artifact_location, run_uuid, "artifacts")
        run_info = RunInfo(run_uuid=run_uuid, experiment_id=experiment_id,
                           name=run_name or '',
                           artifact_uri=artifact_uri, source_type=source_type,
                           source_name=source_name,
                           entry_point_name=entry_point_name, user_id=user_id,
                           status=RunStatus.RUNNING, start_time=start_time, end_time=None,
                           source_version=source_version, lifecycle_stage=LifecycleStage.ACTIVE)
        if self._create_run_info(_entity_to_dict(run_info)):
            for tag in tags:
                self.set_tag(run_uuid, tag)
            if parent_run_id:
                self.set_tag(run_uuid, RunTag(key=MLFLOW_PARENT_RUN_ID, value=parent_run_id))
            if run_name:
                self.set_tag(run_uuid, RunTag(key=MLFLOW_RUN_NAME, value=run_name))
            return Run(run_info=run_info, run_data=None)

    def get_run(self, run_uuid):
        """
        Will get both active and deleted runs.
        """
        _validate_run_id(run_uuid)
        run_info = self._get_run_info(run_uuid)
        metrics = self.get_all_metrics(run_uuid)
        params = self.get_all_params(run_uuid)
        tags = self.get_all_tags(run_uuid)
        return Run(run_info, RunData(metrics, params, tags))

    def _get_run_metrics(self, run_uuid, metric_key=None):
        dynamodb = self._get_dynamodb_resource()
        table_name = '_'.join([self.table_prefix, DynamodbStore.METRICS_TABLE])
        table = dynamodb.Table(table_name)
        condition = Key('run_uuid').eq(run_uuid)
        if metric_key:
            condition = And(condition, Key('key').eq(metric_key))
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
                ReturnConsumedCapacity='TOTAL',
            )
        else:
            response = table.query(
                ConsistentRead=True,
                KeyConditionExpression=condition,
                ReturnConsumedCapacity='TOTAL',
            )
        if response['ResponseMetadata']['HTTPStatusCode'] != 200:
            raise MlflowException("DynamoDB connection error")
        if 'Items' in response:
            return response['Items']
        return []

    def get_metric(self, run_uuid, metric_key):
        _validate_run_id(run_uuid)
        _validate_metric_name(metric_key)
        metrics = self._get_run_metrics(run_uuid, metric_key)
        if not metrics:
            raise MlflowException("Metric '%s' not found under run '%s'" % (metric_key, run_uuid),
                                  RESOURCE_DOES_NOT_EXIST)
        return _list_to_run_metric(metrics)[0]

    def get_all_metrics(self, run_uuid):
        _validate_run_id(run_uuid)
        return _list_to_run_metric(self._get_run_metrics(run_uuid))

    def get_metric_history(self, run_uuid, metric_key):
        _validate_run_id(run_uuid)
        _validate_metric_name(metric_key)
        dynamodb = self._get_dynamodb_resource()
        table_name = '_'.join([self.table_prefix, DynamodbStore.METRICS_TABLE])
        table = dynamodb.Table(table_name)
        response = table.get_item(
            Key={
                'run_uuid': run_uuid,
                'key': metric_key,
            },
            ReturnConsumedCapacity='TOTAL',
        )
        if response['ResponseMetadata']['HTTPStatusCode'] != 200:
            raise MlflowException("DynamoDB connection error")
        if 'Item' in response:
            return _dict_to_run_metric_history(response['Item'])

    def _get_run_params(self, run_uuid, param_name=None):
        dynamodb = self._get_dynamodb_resource()
        table_name = '_'.join([self.table_prefix, DynamodbStore.PARAMS_TABLE])
        table = dynamodb.Table(table_name)
        condition = Key('run_uuid').eq(run_uuid)
        if param_name:
            condition = And(condition, Key('key').eq(param_name))
        response = table.query(
            ConsistentRead=True,
            KeyConditionExpression=condition,
            ReturnConsumedCapacity='TOTAL',
        )
        if response['ResponseMetadata']['HTTPStatusCode'] != 200:
            raise MlflowException("DynamoDB connection error")
        if 'Items' in response:
            return response['Items']
        return []

    def get_param(self, run_uuid, param_name):
        _validate_run_id(run_uuid)
        _validate_param_name(param_name)
        params = self._get_run_params(run_uuid, param_name)
        if not params:
            raise MlflowException("Param '%s' not found under run '%s'" % (param_name, run_uuid),
                                  RESOURCE_DOES_NOT_EXIST)
        return _list_to_run_param(params)[0]

    def get_all_params(self, run_uuid):
        _validate_run_id(run_uuid)
        return _list_to_run_param(self._get_run_params(run_uuid))

    def get_all_tags(self, run_uuid):
        _validate_run_id(run_uuid)
        dynamodb = self._get_dynamodb_resource()
        table_name = '_'.join([self.table_prefix, DynamodbStore.TAGS_TABLE])
        table = dynamodb.Table(table_name)
        response = table.query(
            ConsistentRead=True,
            KeyConditionExpression=Key('run_uuid').eq(run_uuid),
            ReturnConsumedCapacity='TOTAL',
        )
        if response['ResponseMetadata']['HTTPStatusCode'] != 200:
            raise MlflowException("DynamoDB connection error")
        if 'Items' in response:
            return _list_to_run_tag(response['Items'])
        return []

    def _list_runs_uuids(self, experiment_id, view_type=None):
        dynamodb = self._get_dynamodb_resource()
        table_name = '_'.join([self.table_prefix, DynamodbStore.RUN_TABLE])
        table = dynamodb.Table(table_name)

        # Filter on active/deleted with optional experiment_id
        condition = None
        if self.use_gsi and view_type and view_type != ViewType.ALL:
            if view_type == ViewType.ACTIVE_ONLY:
                condition = Key('lifecycle_stage').eq(LifecycleStage.ACTIVE)
            elif view_type == ViewType.DELETED_ONLY:
                condition = Key('lifecycle_stage').eq(LifecycleStage.DELETED)
            if experiment_id:
                condition = And(condition, Key('experiment_id').eq(experiment_id))
            response = table.query(
                IndexName='LifeCycleStage',
                KeyConditionExpression=condition,
                ProjectionExpression='run_uuid, experiment_id, lifecycle_stage',
                ReturnConsumedCapacity='TOTAL',
            )
        elif experiment_id:
            condition = Key('experiment_id').eq(experiment_id)
            response = table.scan(
                FilterExpression=condition,
                ReturnConsumedCapacity='TOTAL',
            )
        else:
            response = table.scan(
                ReturnConsumedCapacity='TOTAL',
            )

        if response['ResponseMetadata']['HTTPStatusCode'] != 200:
            raise MlflowException("DynamoDB connection error")
        if 'Items' in response:
            return _filter_run(response['Items'], view_type, experiment_id)
        return []

    def _get_run_list(self, runs):
        keys = [
            {
                'run_uuid': r['run_uuid']
            } for r in runs
        ]
        if len(keys) == 0:
            return []
        dynamodb = self._get_dynamodb_resource()
        table_name = '_'.join([self.table_prefix, DynamodbStore.RUN_TABLE])
        response = dynamodb.batch_get_item(
            RequestItems={
               table_name: {
                    'Keys': keys,
                }
            },
            ReturnConsumedCapacity='TOTAL'
        )
        if response['ResponseMetadata']['HTTPStatusCode'] != 200:
            raise MlflowException("DynamoDB connection error")
        if 'Responses' in response:
            return response['Responses'][table_name]
        return []

    def search_runs(self, experiment_ids, search_filter, run_view_type):
        matched_runs = []
        for experiment_id in experiment_ids:
            run_uuids = self._list_runs_uuids(experiment_id, run_view_type)
            run_infos = [_dict_to_run_info(r) for r in self._get_run_list(run_uuids)]
            for run_info in run_infos:
                run = Run(run_info=run_info, run_data=None)
                if search_filter.filter(run):
                    matched_runs.append(run)
        return matched_runs

    def list_run_infos(self, experiment_id, run_view_type):
        run_uuids = self._list_runs_uuids(experiment_id, run_view_type)
        return [_dict_to_run_info(r) for r in self._get_run_list(run_uuids)]

    def log_metric(self, run_uuid, metric):
        _validate_run_id(run_uuid)
        _validate_metric_name(metric.key)
        run_info = self._get_run_info(run_uuid)
        check_run_is_active(run_info)
        dynamodb = self._get_dynamodb_resource()
        table_name = '_'.join([self.table_prefix, DynamodbStore.METRICS_TABLE])
        table = dynamodb.Table(table_name)
        # Append metrics to head of list, so the first element is most recent
        response = table.update_item(
            Key={
                'run_uuid': run_uuid,
                'key': metric.key
            },
            UpdateExpression="SET #m = list_append(:m, if_not_exists(#m, :e))",
            ExpressionAttributeNames={'#m': 'metrics'},
            ExpressionAttributeValues={
                 ':e': [
                 ],
                 ':m': [
                    {
                        "value": Decimal(str(metric.value)),
                        "timestamp": metric.timestamp
                    }
                 ],
            },
            ReturnValues="NONE",
            ReturnConsumedCapacity='TOTAL',
        )
        if response['ResponseMetadata']['HTTPStatusCode'] != 200:
            raise MlflowException("DynamoDB connection error")
        return True

    def log_param(self, run_uuid, param):
        _validate_run_id(run_uuid)
        _validate_param_name(param.key)
        run_info = self._get_run_info(run_uuid)
        check_run_is_active(run_info)
        dynamodb = self._get_dynamodb_resource()
        table_name = '_'.join([self.table_prefix, DynamodbStore.PARAMS_TABLE])
        table = dynamodb.Table(table_name)
        response = table.put_item(
            Item={
                'run_uuid': run_uuid,
                'key': param.key,
                'value': param.value,
            },
            ReturnConsumedCapacity='TOTAL',
        )
        if response['ResponseMetadata']['HTTPStatusCode'] != 200:
            raise MlflowException("DynamoDB connection error")
        return True

    def set_tag(self, run_uuid, tag):
        _validate_run_id(run_uuid)
        _validate_tag_name(tag.key)
        run_info = self._get_run_info(run_uuid)
        check_run_is_active(run_info)
        dynamodb = self._get_dynamodb_resource()
        table_name = '_'.join([self.table_prefix, DynamodbStore.TAGS_TABLE])
        table = dynamodb.Table(table_name)
        response = table.put_item(
            Item={
                'run_uuid': run_uuid,
                'key': tag.key,
                'value': tag.value,
            },
            ReturnConsumedCapacity='TOTAL',
        )
        if response['ResponseMetadata']['HTTPStatusCode'] != 200:
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
