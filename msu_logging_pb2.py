# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# NO CHECKED-IN PROTOBUF GENCODE
# source: msu_logging.proto
# Protobuf Python Version: 5.29.0
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import runtime_version as _runtime_version
from google.protobuf import symbol_database as _symbol_database
from google.protobuf.internal import builder as _builder
_runtime_version.ValidateProtobufRuntimeVersion(
    _runtime_version.Domain.PUBLIC,
    5,
    29,
    0,
    '',
    'msu_logging.proto'
)
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\x11msu_logging.proto\x12\x0bmsu_logging\"Y\n\x10TranscribeResult\x12\x0f\n\x07success\x18\x01 \x01(\x08\x12\x14\n\x0c\x65rrorMessage\x18\x02 \x01(\t\x12\x0e\n\x06result\x18\x03 \x01(\t\x12\x0e\n\x06taskId\x18\x04 \x01(\x05\"W\n\x0eProtocolResult\x12\x0f\n\x07success\x18\x01 \x01(\x08\x12\x14\n\x0c\x65rrorMessage\x18\x02 \x01(\t\x12\x0e\n\x06result\x18\x03 \x01(\t\x12\x0e\n\x06taskId\x18\x05 \x01(\x05\"/\n\x06Result\x12\x0f\n\x07success\x18\x01 \x01(\x08\x12\x14\n\x0c\x65rrorMessage\x18\x02 \x01(\t2X\n\nTranscribe\x12J\n\x14SendTranscribeResult\x12\x1d.msu_logging.TranscribeResult\x1a\x13.msu_logging.Result2R\n\x08Protocol\x12\x46\n\x12SendProtocolResult\x12\x1b.msu_logging.ProtocolResult\x1a\x13.msu_logging.ResultB-Z+makarmolochaev.msu_logging.v1;msu_loggingv1b\x06proto3')

_globals = globals()
_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, _globals)
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'msu_logging_pb2', _globals)
if not _descriptor._USE_C_DESCRIPTORS:
  _globals['DESCRIPTOR']._loaded_options = None
  _globals['DESCRIPTOR']._serialized_options = b'Z+makarmolochaev.msu_logging.v1;msu_loggingv1'
  _globals['_TRANSCRIBERESULT']._serialized_start=34
  _globals['_TRANSCRIBERESULT']._serialized_end=123
  _globals['_PROTOCOLRESULT']._serialized_start=125
  _globals['_PROTOCOLRESULT']._serialized_end=212
  _globals['_RESULT']._serialized_start=214
  _globals['_RESULT']._serialized_end=261
  _globals['_TRANSCRIBE']._serialized_start=263
  _globals['_TRANSCRIBE']._serialized_end=351
  _globals['_PROTOCOL']._serialized_start=353
  _globals['_PROTOCOL']._serialized_end=435
# @@protoc_insertion_point(module_scope)
