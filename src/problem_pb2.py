# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: problem.proto

import sys
_b=sys.version_info[0]<3 and (lambda x:x) or (lambda x:x.encode('latin1'))
from google.protobuf.internal import enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


from google.protobuf import descriptor_pb2 as google_dot_protobuf_dot_descriptor__pb2


DESCRIPTOR = _descriptor.FileDescriptor(
  name='problem.proto',
  package='',
  syntax='proto3',
  serialized_options=_b('Z\010pipeline'),
  serialized_pb=_b('\n\rproblem.proto\x1a google/protobuf/descriptor.proto\"\\\n\x18ProblemPerformanceMetric\x12\"\n\x06metric\x18\x01 \x01(\x0e\x32\x12.PerformanceMetric\x12\t\n\x01k\x18\x02 \x01(\x05\x12\x11\n\tpos_label\x18\x03 \x01(\t\"\xd3\x01\n\x07Problem\x12\x0e\n\x02id\x18\x01 \x01(\tB\x02\x18\x01\x12\x13\n\x07version\x18\x02 \x01(\tB\x02\x18\x01\x12\x10\n\x04name\x18\x03 \x01(\tB\x02\x18\x01\x12\x17\n\x0b\x64\x65scription\x18\x04 \x01(\tB\x02\x18\x01\x12\x1c\n\ttask_type\x18\x05 \x01(\x0e\x32\t.TaskType\x12\"\n\x0ctask_subtype\x18\x06 \x01(\x0e\x32\x0c.TaskSubtype\x12\x36\n\x13performance_metrics\x18\x07 \x03(\x0b\x32\x19.ProblemPerformanceMetric\"~\n\rProblemTarget\x12\x14\n\x0ctarget_index\x18\x01 \x01(\x05\x12\x13\n\x0bresource_id\x18\x02 \x01(\t\x12\x14\n\x0c\x63olumn_index\x18\x03 \x01(\x05\x12\x13\n\x0b\x63olumn_name\x18\x04 \x01(\t\x12\x17\n\x0f\x63lusters_number\x18\x05 \x01(\x05\"C\n\x0cProblemInput\x12\x12\n\ndataset_id\x18\x01 \x01(\t\x12\x1f\n\x07targets\x18\x02 \x03(\x0b\x32\x0e.ProblemTarget\"4\n\x10\x44\x61taAugmentation\x12\x0e\n\x06\x64omain\x18\x01 \x03(\t\x12\x10\n\x08keywords\x18\x02 \x03(\t\"\xcc\x01\n\x12ProblemDescription\x12\x19\n\x07problem\x18\x01 \x01(\x0b\x32\x08.Problem\x12\x1d\n\x06inputs\x18\x02 \x03(\x0b\x32\r.ProblemInput\x12\n\n\x02id\x18\x03 \x01(\t\x12\x0f\n\x07version\x18\x04 \x01(\t\x12\x0c\n\x04name\x18\x05 \x01(\t\x12\x13\n\x0b\x64\x65scription\x18\x06 \x01(\t\x12\x0e\n\x06\x64igest\x18\x07 \x01(\t\x12,\n\x11\x64\x61ta_augmentation\x18\x08 \x03(\x0b\x32\x11.DataAugmentation*\x96\x02\n\x08TaskType\x12\x17\n\x13TASK_TYPE_UNDEFINED\x10\x00\x12\x12\n\x0e\x43LASSIFICATION\x10\x01\x12\x0e\n\nREGRESSION\x10\x02\x12\x0e\n\nCLUSTERING\x10\x03\x12\x13\n\x0fLINK_PREDICTION\x10\x04\x12\x15\n\x11VERTEX_NOMINATION\x10\x05\x12\x17\n\x13\x43OMMUNITY_DETECTION\x10\x06\x12\x14\n\x10GRAPH_CLUSTERING\x10\x07\x12\x12\n\x0eGRAPH_MATCHING\x10\x08\x12\x1b\n\x17TIME_SERIES_FORECASTING\x10\t\x12\x1b\n\x17\x43OLLABORATIVE_FILTERING\x10\n\x12\x14\n\x10OBJECT_DETECTION\x10\x0b*\xa6\x01\n\x0bTaskSubtype\x12\x1a\n\x16TASK_SUBTYPE_UNDEFINED\x10\x00\x12\x08\n\x04NONE\x10\x01\x12\n\n\x06\x42INARY\x10\x02\x12\x0e\n\nMULTICLASS\x10\x03\x12\x0e\n\nMULTILABEL\x10\x04\x12\x0e\n\nUNIVARIATE\x10\x05\x12\x10\n\x0cMULTIVARIATE\x10\x06\x12\x0f\n\x0bOVERLAPPING\x10\x07\x12\x12\n\x0eNONOVERLAPPING\x10\x08*\xb2\x03\n\x11PerformanceMetric\x12\x14\n\x10METRIC_UNDEFINED\x10\x00\x12\x0c\n\x08\x41\x43\x43URACY\x10\x01\x12\r\n\tPRECISION\x10\x02\x12\n\n\x06RECALL\x10\x03\x12\x06\n\x02\x46\x31\x10\x04\x12\x0c\n\x08\x46\x31_MICRO\x10\x05\x12\x0c\n\x08\x46\x31_MACRO\x10\x06\x12\x0b\n\x07ROC_AUC\x10\x07\x12\x11\n\rROC_AUC_MICRO\x10\x08\x12\x11\n\rROC_AUC_MACRO\x10\t\x12\x16\n\x12MEAN_SQUARED_ERROR\x10\n\x12\x1b\n\x17ROOT_MEAN_SQUARED_ERROR\x10\x0b\x12\x1f\n\x1bROOT_MEAN_SQUARED_ERROR_AVG\x10\x0c\x12\x17\n\x13MEAN_ABSOLUTE_ERROR\x10\r\x12\r\n\tR_SQUARED\x10\x0e\x12!\n\x1dNORMALIZED_MUTUAL_INFORMATION\x10\x0f\x12\x1c\n\x18JACCARD_SIMILARITY_SCORE\x10\x10\x12\x16\n\x12PRECISION_AT_TOP_K\x10\x11\x12&\n\"OBJECT_DETECTION_AVERAGE_PRECISION\x10\x12\x12\x08\n\x04LOSS\x10\x64\x42\nZ\x08pipelineb\x06proto3')
  ,
  dependencies=[google_dot_protobuf_dot_descriptor__pb2.DESCRIPTOR,])

_TASKTYPE = _descriptor.EnumDescriptor(
  name='TaskType',
  full_name='TaskType',
  filename=None,
  file=DESCRIPTOR,
  values=[
    _descriptor.EnumValueDescriptor(
      name='TASK_TYPE_UNDEFINED', index=0, number=0,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='CLASSIFICATION', index=1, number=1,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='REGRESSION', index=2, number=2,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='CLUSTERING', index=3, number=3,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='LINK_PREDICTION', index=4, number=4,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='VERTEX_NOMINATION', index=5, number=5,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='COMMUNITY_DETECTION', index=6, number=6,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='GRAPH_CLUSTERING', index=7, number=7,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='GRAPH_MATCHING', index=8, number=8,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='TIME_SERIES_FORECASTING', index=9, number=9,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='COLLABORATIVE_FILTERING', index=10, number=10,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='OBJECT_DETECTION', index=11, number=11,
      serialized_options=None,
      type=None),
  ],
  containing_type=None,
  serialized_options=None,
  serialized_start=818,
  serialized_end=1096,
)
_sym_db.RegisterEnumDescriptor(_TASKTYPE)

TaskType = enum_type_wrapper.EnumTypeWrapper(_TASKTYPE)
_TASKSUBTYPE = _descriptor.EnumDescriptor(
  name='TaskSubtype',
  full_name='TaskSubtype',
  filename=None,
  file=DESCRIPTOR,
  values=[
    _descriptor.EnumValueDescriptor(
      name='TASK_SUBTYPE_UNDEFINED', index=0, number=0,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='NONE', index=1, number=1,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='BINARY', index=2, number=2,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='MULTICLASS', index=3, number=3,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='MULTILABEL', index=4, number=4,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='UNIVARIATE', index=5, number=5,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='MULTIVARIATE', index=6, number=6,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='OVERLAPPING', index=7, number=7,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='NONOVERLAPPING', index=8, number=8,
      serialized_options=None,
      type=None),
  ],
  containing_type=None,
  serialized_options=None,
  serialized_start=1099,
  serialized_end=1265,
)
_sym_db.RegisterEnumDescriptor(_TASKSUBTYPE)

TaskSubtype = enum_type_wrapper.EnumTypeWrapper(_TASKSUBTYPE)
_PERFORMANCEMETRIC = _descriptor.EnumDescriptor(
  name='PerformanceMetric',
  full_name='PerformanceMetric',
  filename=None,
  file=DESCRIPTOR,
  values=[
    _descriptor.EnumValueDescriptor(
      name='METRIC_UNDEFINED', index=0, number=0,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='ACCURACY', index=1, number=1,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='PRECISION', index=2, number=2,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='RECALL', index=3, number=3,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='F1', index=4, number=4,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='F1_MICRO', index=5, number=5,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='F1_MACRO', index=6, number=6,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='ROC_AUC', index=7, number=7,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='ROC_AUC_MICRO', index=8, number=8,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='ROC_AUC_MACRO', index=9, number=9,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='MEAN_SQUARED_ERROR', index=10, number=10,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='ROOT_MEAN_SQUARED_ERROR', index=11, number=11,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='ROOT_MEAN_SQUARED_ERROR_AVG', index=12, number=12,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='MEAN_ABSOLUTE_ERROR', index=13, number=13,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='R_SQUARED', index=14, number=14,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='NORMALIZED_MUTUAL_INFORMATION', index=15, number=15,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='JACCARD_SIMILARITY_SCORE', index=16, number=16,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='PRECISION_AT_TOP_K', index=17, number=17,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='OBJECT_DETECTION_AVERAGE_PRECISION', index=18, number=18,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='LOSS', index=19, number=100,
      serialized_options=None,
      type=None),
  ],
  containing_type=None,
  serialized_options=None,
  serialized_start=1268,
  serialized_end=1702,
)
_sym_db.RegisterEnumDescriptor(_PERFORMANCEMETRIC)

PerformanceMetric = enum_type_wrapper.EnumTypeWrapper(_PERFORMANCEMETRIC)
TASK_TYPE_UNDEFINED = 0
CLASSIFICATION = 1
REGRESSION = 2
CLUSTERING = 3
LINK_PREDICTION = 4
VERTEX_NOMINATION = 5
COMMUNITY_DETECTION = 6
GRAPH_CLUSTERING = 7
GRAPH_MATCHING = 8
TIME_SERIES_FORECASTING = 9
COLLABORATIVE_FILTERING = 10
OBJECT_DETECTION = 11
TASK_SUBTYPE_UNDEFINED = 0
NONE = 1
BINARY = 2
MULTICLASS = 3
MULTILABEL = 4
UNIVARIATE = 5
MULTIVARIATE = 6
OVERLAPPING = 7
NONOVERLAPPING = 8
METRIC_UNDEFINED = 0
ACCURACY = 1
PRECISION = 2
RECALL = 3
F1 = 4
F1_MICRO = 5
F1_MACRO = 6
ROC_AUC = 7
ROC_AUC_MICRO = 8
ROC_AUC_MACRO = 9
MEAN_SQUARED_ERROR = 10
ROOT_MEAN_SQUARED_ERROR = 11
ROOT_MEAN_SQUARED_ERROR_AVG = 12
MEAN_ABSOLUTE_ERROR = 13
R_SQUARED = 14
NORMALIZED_MUTUAL_INFORMATION = 15
JACCARD_SIMILARITY_SCORE = 16
PRECISION_AT_TOP_K = 17
OBJECT_DETECTION_AVERAGE_PRECISION = 18
LOSS = 100



_PROBLEMPERFORMANCEMETRIC = _descriptor.Descriptor(
  name='ProblemPerformanceMetric',
  full_name='ProblemPerformanceMetric',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='metric', full_name='ProblemPerformanceMetric.metric', index=0,
      number=1, type=14, cpp_type=8, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='k', full_name='ProblemPerformanceMetric.k', index=1,
      number=2, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='pos_label', full_name='ProblemPerformanceMetric.pos_label', index=2,
      number=3, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=51,
  serialized_end=143,
)


_PROBLEM = _descriptor.Descriptor(
  name='Problem',
  full_name='Problem',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='id', full_name='Problem.id', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=_b('\030\001'), file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='version', full_name='Problem.version', index=1,
      number=2, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=_b('\030\001'), file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='name', full_name='Problem.name', index=2,
      number=3, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=_b('\030\001'), file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='description', full_name='Problem.description', index=3,
      number=4, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=_b('\030\001'), file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='task_type', full_name='Problem.task_type', index=4,
      number=5, type=14, cpp_type=8, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='task_subtype', full_name='Problem.task_subtype', index=5,
      number=6, type=14, cpp_type=8, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='performance_metrics', full_name='Problem.performance_metrics', index=6,
      number=7, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=146,
  serialized_end=357,
)


_PROBLEMTARGET = _descriptor.Descriptor(
  name='ProblemTarget',
  full_name='ProblemTarget',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='target_index', full_name='ProblemTarget.target_index', index=0,
      number=1, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='resource_id', full_name='ProblemTarget.resource_id', index=1,
      number=2, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='column_index', full_name='ProblemTarget.column_index', index=2,
      number=3, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='column_name', full_name='ProblemTarget.column_name', index=3,
      number=4, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='clusters_number', full_name='ProblemTarget.clusters_number', index=4,
      number=5, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=359,
  serialized_end=485,
)


_PROBLEMINPUT = _descriptor.Descriptor(
  name='ProblemInput',
  full_name='ProblemInput',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='dataset_id', full_name='ProblemInput.dataset_id', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='targets', full_name='ProblemInput.targets', index=1,
      number=2, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=487,
  serialized_end=554,
)


_DATAAUGMENTATION = _descriptor.Descriptor(
  name='DataAugmentation',
  full_name='DataAugmentation',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='domain', full_name='DataAugmentation.domain', index=0,
      number=1, type=9, cpp_type=9, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='keywords', full_name='DataAugmentation.keywords', index=1,
      number=2, type=9, cpp_type=9, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=556,
  serialized_end=608,
)


_PROBLEMDESCRIPTION = _descriptor.Descriptor(
  name='ProblemDescription',
  full_name='ProblemDescription',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='problem', full_name='ProblemDescription.problem', index=0,
      number=1, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='inputs', full_name='ProblemDescription.inputs', index=1,
      number=2, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='id', full_name='ProblemDescription.id', index=2,
      number=3, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='version', full_name='ProblemDescription.version', index=3,
      number=4, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='name', full_name='ProblemDescription.name', index=4,
      number=5, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='description', full_name='ProblemDescription.description', index=5,
      number=6, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='digest', full_name='ProblemDescription.digest', index=6,
      number=7, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='data_augmentation', full_name='ProblemDescription.data_augmentation', index=7,
      number=8, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=611,
  serialized_end=815,
)

_PROBLEMPERFORMANCEMETRIC.fields_by_name['metric'].enum_type = _PERFORMANCEMETRIC
_PROBLEM.fields_by_name['task_type'].enum_type = _TASKTYPE
_PROBLEM.fields_by_name['task_subtype'].enum_type = _TASKSUBTYPE
_PROBLEM.fields_by_name['performance_metrics'].message_type = _PROBLEMPERFORMANCEMETRIC
_PROBLEMINPUT.fields_by_name['targets'].message_type = _PROBLEMTARGET
_PROBLEMDESCRIPTION.fields_by_name['problem'].message_type = _PROBLEM
_PROBLEMDESCRIPTION.fields_by_name['inputs'].message_type = _PROBLEMINPUT
_PROBLEMDESCRIPTION.fields_by_name['data_augmentation'].message_type = _DATAAUGMENTATION
DESCRIPTOR.message_types_by_name['ProblemPerformanceMetric'] = _PROBLEMPERFORMANCEMETRIC
DESCRIPTOR.message_types_by_name['Problem'] = _PROBLEM
DESCRIPTOR.message_types_by_name['ProblemTarget'] = _PROBLEMTARGET
DESCRIPTOR.message_types_by_name['ProblemInput'] = _PROBLEMINPUT
DESCRIPTOR.message_types_by_name['DataAugmentation'] = _DATAAUGMENTATION
DESCRIPTOR.message_types_by_name['ProblemDescription'] = _PROBLEMDESCRIPTION
DESCRIPTOR.enum_types_by_name['TaskType'] = _TASKTYPE
DESCRIPTOR.enum_types_by_name['TaskSubtype'] = _TASKSUBTYPE
DESCRIPTOR.enum_types_by_name['PerformanceMetric'] = _PERFORMANCEMETRIC
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

ProblemPerformanceMetric = _reflection.GeneratedProtocolMessageType('ProblemPerformanceMetric', (_message.Message,), dict(
  DESCRIPTOR = _PROBLEMPERFORMANCEMETRIC,
  __module__ = 'problem_pb2'
  # @@protoc_insertion_point(class_scope:ProblemPerformanceMetric)
  ))
_sym_db.RegisterMessage(ProblemPerformanceMetric)

Problem = _reflection.GeneratedProtocolMessageType('Problem', (_message.Message,), dict(
  DESCRIPTOR = _PROBLEM,
  __module__ = 'problem_pb2'
  # @@protoc_insertion_point(class_scope:Problem)
  ))
_sym_db.RegisterMessage(Problem)

ProblemTarget = _reflection.GeneratedProtocolMessageType('ProblemTarget', (_message.Message,), dict(
  DESCRIPTOR = _PROBLEMTARGET,
  __module__ = 'problem_pb2'
  # @@protoc_insertion_point(class_scope:ProblemTarget)
  ))
_sym_db.RegisterMessage(ProblemTarget)

ProblemInput = _reflection.GeneratedProtocolMessageType('ProblemInput', (_message.Message,), dict(
  DESCRIPTOR = _PROBLEMINPUT,
  __module__ = 'problem_pb2'
  # @@protoc_insertion_point(class_scope:ProblemInput)
  ))
_sym_db.RegisterMessage(ProblemInput)

DataAugmentation = _reflection.GeneratedProtocolMessageType('DataAugmentation', (_message.Message,), dict(
  DESCRIPTOR = _DATAAUGMENTATION,
  __module__ = 'problem_pb2'
  # @@protoc_insertion_point(class_scope:DataAugmentation)
  ))
_sym_db.RegisterMessage(DataAugmentation)

ProblemDescription = _reflection.GeneratedProtocolMessageType('ProblemDescription', (_message.Message,), dict(
  DESCRIPTOR = _PROBLEMDESCRIPTION,
  __module__ = 'problem_pb2'
  # @@protoc_insertion_point(class_scope:ProblemDescription)
  ))
_sym_db.RegisterMessage(ProblemDescription)


DESCRIPTOR._options = None
_PROBLEM.fields_by_name['id']._options = None
_PROBLEM.fields_by_name['version']._options = None
_PROBLEM.fields_by_name['name']._options = None
_PROBLEM.fields_by_name['description']._options = None
# @@protoc_insertion_point(module_scope)
