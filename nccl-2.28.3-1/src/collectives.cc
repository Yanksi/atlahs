/*************************************************************************
 * Copyright (c) 2015-2023, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "argcheck.h" // Need some checks here since we access comm
#include "collectives.h"
#include "enqueue.h"
#include "nccl.h"
#include "nvtx_payload_schemas.h"

#if defined(ENABLE_API_NVTX)
#include "nvtx3/nvToolsExt.h"
#endif

#if defined(ENABLE_API_NVTX)
  const uint32_t colors[] = {
    0xffe91e63, 
    0xff2196f3, 
    0xff4caf50, 
    0xffffc107, 
    0xff9c27b0, 
    0xffff5722, 
    0xff00bcd4, 
    0xff673ab7, 
    0xffff9800, 
    0xff03a9f4};
  nvtxEventAttributes_t eventAttrib_allgather = {0};  
  nvtxEventAttributes_t eventAttrib_allreduce = {0};
  nvtxEventAttributes_t eventAttrib_broadcast = {0};
  nvtxEventAttributes_t eventAttrib_reduce = {0};
  nvtxEventAttributes_t eventAttrib_reducescatter = {0};
  nvtxEventAttributes_t eventAttrib_send = {0};
  nvtxEventAttributes_t eventAttrib_recv = {0};
#endif

const char* ncclFuncToString(ncclFunc_t fn) {
  switch (fn) {
  case ncclFuncAllGather: return "AllGather";
  case ncclFuncAllReduce: return "AllReduce";
  case ncclFuncAlltoAll: return "AlltoAll";
  case ncclFuncBroadcast: return "Broadcast";
  case ncclFuncGather: return "Gather";
  case ncclFuncRecv: return "Recv";
  case ncclFuncReduce: return "Reduce";
  case ncclFuncReduceScatter: return "ReduceScatter";
  case ncclFuncScatter: return "Scatter";
  case ncclFuncSendRecv: return "SendRecv";
  case ncclFuncSend: return "Send";
  default: return "Invalid";
  }
}

const char* ncclDevRedOpToString(ncclDevRedOp_t op) {
  switch (op) {
  case ncclDevSum: return "Sum";
  case ncclDevProd: return "Prod";
  case ncclDevMinMax: return "MinMax";
  case ncclDevPreMulSum: return "PreMulSum";
  case ncclDevSumPostDiv: return "SumPostDiv";
  default: return "Unknown";
  }
}

const char* ncclDatatypeToString(ncclDataType_t type) {
  switch (type) {
  case ncclInt8: return "ncclInt8";
  case ncclInt32: return "ncclInt32";
  case ncclUint32: return "ncclUint32";
  case ncclInt64: return "ncclInt64";
  case ncclUint64: return "ncclUint64";
  case ncclFloat16: return "ncclFloat16";
  case ncclFloat32: return "ncclFloat32";
  case ncclFloat64: return "ncclFloat64";
  case ncclBfloat16: return "ncclBfloat16";
  case ncclFloat8e4m3: return "ncclFloat8e4m3";
  case ncclFloat8e5m2: return "ncclFloat8e5m2";
  default: return "Unknown";
  }
}

const char* ncclAlgoToString(int algo) {
  switch (algo) {
  case NCCL_ALGO_TREE: return "TREE";
  case NCCL_ALGO_RING: return "RING";
  case NCCL_ALGO_COLLNET_DIRECT: return "COLLNET_DIRECT";
  case NCCL_ALGO_COLLNET_CHAIN: return "COLLNET_CHAIN";
  case NCCL_ALGO_NVLS: return "NVLS";
  case NCCL_ALGO_NVLS_TREE: return "NVLS_TREE";
  case NCCL_ALGO_PAT: return "PAT";
  default: return "Unknown";
  }
}

const char* ncclProtoToString(int proto) {
  switch (proto) {
  case NCCL_PROTO_LL: return "LL";
  case NCCL_PROTO_LL128: return "LL128";
  case NCCL_PROTO_SIMPLE: return "SIMPLE";
  default: return "Unknown";
  }
}

NCCL_API(ncclResult_t, ncclAllGather, const void* sendbuff, void* recvbuff, size_t sendcount,
    ncclDataType_t datatype, ncclComm_t comm, cudaStream_t stream);
ncclResult_t ncclAllGather(const void* sendbuff, void* recvbuff, size_t sendcount,
    ncclDataType_t datatype, ncclComm_t comm, cudaStream_t stream) {


  #if defined(ENABLE_API_NVTX)
  char nvtxMsg_AllGather[256];
  pid_t pid = getpid();
  snprintf(nvtxMsg_AllGather, sizeof(nvtxMsg_AllGather), 
                  "ncclAllGather(): commHash 0x%llx, stream %p, data_size %zu, type_size %d, pid %d", 
                  (unsigned long long)comm->commHash, 
                  stream, 
                  sendcount * ncclTypeSize(datatype),
                  ncclTypeSize(datatype),
                  pid);

  eventAttrib_allgather.version = NVTX_VERSION;
  eventAttrib_allgather.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
  eventAttrib_allgather.messageType = NVTX_MESSAGE_TYPE_ASCII;
  eventAttrib_allgather.colorType = NVTX_COLOR_ARGB;
  eventAttrib_allgather.message.ascii = nvtxMsg_AllGather;
  eventAttrib_allgather.color = colors[0];

  nvtxRangePushEx(&eventAttrib_allgather);
  #endif  

  // Just pass the size of one message and not the total bytes sent/received.
  NVTX3_FUNC_WITH_PARAMS(AllGather, NcclNvtxParamsAllGather,
    NVTX3_PAYLOAD(comm ? comm->commHash : 0, sendcount * ncclTypeSize(datatype)));

  struct ncclInfo info = { ncclFuncAllGather, "AllGather",
    sendbuff, recvbuff, sendcount, datatype, ncclSum, 0, comm, stream, /* Args */
    ALLGATHER_CHUNKSTEPS, ALLGATHER_SLICESTEPS };

  ncclResult_t ret;
  ret = ncclEnqueueCheck(&info);

  #if defined(ENABLE_API_NVTX)
    nvtxRangePop();
  #endif

  return ret;
}


// TO DO
NCCL_API(ncclResult_t, ncclAlltoAll, const void* sendbuff, void* recvbuff, size_t count,
    ncclDataType_t datatype, ncclComm* comm, cudaStream_t stream);
ncclResult_t ncclAlltoAll(const void* sendbuff, void* recvbuff, size_t count,
    ncclDataType_t datatype, ncclComm* comm, cudaStream_t stream) {
  NVTX3_FUNC_WITH_PARAMS(AlltoAll, NcclNvtxParamsAlltoAll,
    NVTX3_PAYLOAD(comm ? comm->commHash : 0, count * ncclTypeSize(datatype)));

  struct ncclInfo info = { ncclFuncAlltoAll, "AlltoAll",
    sendbuff, recvbuff, count, datatype, ncclSum, 0, comm, stream, /* Args */
    ALLTOALL_CHUNKSTEPS, ALLTOALL_SLICESTEPS };
  return ncclEnqueueCheck(&info);
}

NCCL_API(ncclResult_t, ncclAllReduce, const void* sendbuff, void* recvbuff, size_t count,
    ncclDataType_t datatype, ncclRedOp_t op, ncclComm* comm, cudaStream_t stream);
ncclResult_t ncclAllReduce(const void* sendbuff, void* recvbuff, size_t count,
    ncclDataType_t datatype, ncclRedOp_t op, ncclComm* comm, cudaStream_t stream) {


#if defined(ENABLE_API_NVTX)
  char nvtxMsg_AllReduce[256];
  pid_t pid = getpid();
  snprintf(nvtxMsg_AllReduce, sizeof(nvtxMsg_AllReduce), 
                  "ncclAllReduce(): commHash 0x%llx, stream %p, data_size %zu, type_size %d, red_op %d, pid %d", 
                  (unsigned long long)comm->commHash, 
                  stream, 
                  count * ncclTypeSize(datatype),
                  ncclTypeSize(datatype),
                  op,
                  pid);

  eventAttrib_allreduce.version = NVTX_VERSION;
  eventAttrib_allreduce.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
  eventAttrib_allreduce.messageType = NVTX_MESSAGE_TYPE_ASCII;
  eventAttrib_allreduce.colorType = NVTX_COLOR_ARGB;
  eventAttrib_allreduce.message.ascii = nvtxMsg_AllReduce;
  eventAttrib_allreduce.color = colors[1];

  nvtxRangePushEx(&eventAttrib_allreduce);
#endif

  NVTX3_FUNC_WITH_PARAMS(AllReduce, NcclNvtxParamsAllReduce,
    NVTX3_PAYLOAD(comm ? comm->commHash : 0, count * ncclTypeSize(datatype), op));

  struct ncclInfo info = { ncclFuncAllReduce, "AllReduce",
    sendbuff, recvbuff, count, datatype, op, 0, comm, stream, /* Args */
    ALLREDUCE_CHUNKSTEPS, ALLREDUCE_SLICESTEPS };
  
    ncclResult_t ret;
  ret = ncclEnqueueCheck(&info);

#if defined(ENABLE_API_NVTX)
  nvtxRangePop();
#endif

  return ret;
}

NCCL_API(ncclResult_t, ncclBroadcast, const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype, int root,
    ncclComm_t comm, cudaStream_t stream);
ncclResult_t ncclBroadcast(const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype, int root,
    ncclComm_t comm, cudaStream_t stream) {

#if defined(ENABLE_API_NVTX)
  char nvtxMsg_Broadcast[256];
  pid_t pid = getpid();
  snprintf(nvtxMsg_Broadcast, sizeof(nvtxMsg_Broadcast), 
                  "ncclBroadcast(): commHash 0x%llx, stream %p, data_size %zu, type_size %d, root %d, pid %d", 
                  (unsigned long long)comm->commHash, 
                  stream, 
                  count * ncclTypeSize(datatype),
                  ncclTypeSize(datatype), 
                  root,
                  pid);

  eventAttrib_broadcast.version = NVTX_VERSION;
  eventAttrib_broadcast.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
  eventAttrib_broadcast.messageType = NVTX_MESSAGE_TYPE_ASCII;
  eventAttrib_broadcast.colorType = NVTX_COLOR_ARGB;
  eventAttrib_broadcast.message.ascii = nvtxMsg_Broadcast;
  eventAttrib_broadcast.color = colors[2];

  nvtxRangePushEx(&eventAttrib_broadcast);
#endif


  NVTX3_FUNC_WITH_PARAMS(Broadcast, NcclNvtxParamsBroadcast,
    NVTX3_PAYLOAD(comm ? comm->commHash : 0, count * ncclTypeSize(datatype), root));

  struct ncclInfo info = { ncclFuncBroadcast, "Broadcast",
    sendbuff, recvbuff, count, datatype, ncclSum, root, comm, stream, /* Args */
    BROADCAST_CHUNKSTEPS, BROADCAST_SLICESTEPS };
  
  ncclResult_t ret;
  ret = ncclEnqueueCheck(&info);

#if defined(ENABLE_API_NVTX)
  nvtxRangePop();
#endif
  
  return ret;
}
/* Deprecated original "in place" function, similar to MPI */
NCCL_API(ncclResult_t, ncclBcast, void* buff, size_t count, ncclDataType_t datatype, int root,
    ncclComm_t comm, cudaStream_t stream);
ncclResult_t ncclBcast(void* buff, size_t count, ncclDataType_t datatype, int root,
    ncclComm_t comm, cudaStream_t stream) {
  return ncclBroadcast(buff, buff, count, datatype, root, comm, stream);
}

NCCL_API(ncclResult_t, ncclGather, const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype, int root,
    ncclComm* comm, cudaStream_t stream);
ncclResult_t ncclGather(const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype, int root,
    ncclComm* comm, cudaStream_t stream) {
  NVTX3_FUNC_WITH_PARAMS(Gather, NcclNvtxParamsGather,
    NVTX3_PAYLOAD(comm ? comm->commHash : 0, count * ncclTypeSize(datatype), root));

  struct ncclInfo info = { ncclFuncGather, "Gather",
    sendbuff, recvbuff, count, datatype, ncclSum, root, comm, stream, /* Args */
    GATHER_CHUNKSTEPS, GATHER_SLICESTEPS };
  return ncclEnqueueCheck(&info);
}

NCCL_API(ncclResult_t, ncclReduce, const void* sendbuff, void* recvbuff, size_t count,
    ncclDataType_t datatype, ncclRedOp_t op, int root, ncclComm_t comm, cudaStream_t stream);
ncclResult_t ncclReduce(const void* sendbuff, void* recvbuff, size_t count,
    ncclDataType_t datatype, ncclRedOp_t op, int root, ncclComm_t comm, cudaStream_t stream) {

#if defined(ENABLE_API_NVTX)
  char nvtxMsg_Reduce[256];
  pid_t pid = getpid();
  snprintf(nvtxMsg_Reduce, sizeof(nvtxMsg_Reduce), 
                  "ncclReduce(): commHash 0x%llx, stream %p, data_size %zu, type_size %d, red_op %d, root %d, pid %d", 
                  (unsigned long long)comm->commHash, 
                  stream, 
                  count * ncclTypeSize(datatype),
                  ncclTypeSize(datatype), 
                  op,
                  root,
                  pid);

  eventAttrib_reduce.version = NVTX_VERSION;
  eventAttrib_reduce.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
  eventAttrib_reduce.messageType = NVTX_MESSAGE_TYPE_ASCII;
  eventAttrib_reduce.colorType = NVTX_COLOR_ARGB;
  eventAttrib_reduce.message.ascii = nvtxMsg_Reduce;
  eventAttrib_reduce.color = colors[4];

  nvtxRangePushEx(&eventAttrib_reduce);
#endif

  NVTX3_FUNC_WITH_PARAMS(Reduce, NcclNvtxParamsReduce,
    NVTX3_PAYLOAD(comm ? comm->commHash : 0, count * ncclTypeSize(datatype), root, op));

  struct ncclInfo info = { ncclFuncReduce, "Reduce",
    sendbuff, recvbuff, count, datatype, op, root, comm, stream, /* Args */
    REDUCE_CHUNKSTEPS, REDUCE_SLICESTEPS };
  
  ncclResult_t ret;
  ret = ncclEnqueueCheck(&info);

#if defined(ENABLE_API_NVTX)
  nvtxRangePop();
#endif

  return ret;
}

NCCL_API(ncclResult_t, ncclReduceScatter, const void* sendbuff, void* recvbuff, size_t recvcount,
    ncclDataType_t datatype, ncclRedOp_t op, ncclComm* comm, cudaStream_t stream);
ncclResult_t ncclReduceScatter(const void* sendbuff, void* recvbuff, size_t recvcount,
    ncclDataType_t datatype, ncclRedOp_t op, ncclComm* comm, cudaStream_t stream) {

#if defined(ENABLE_API_NVTX)
  char nvtxMsg_ReduceScatter[256];
  pid_t pid = getpid();
  snprintf(nvtxMsg_ReduceScatter, sizeof(nvtxMsg_ReduceScatter), 
                  "ncclReduceScatter(): commHash 0x%llx, stream %p, data_size %zu, type_size %d, red_op %d, pid %d", 
                  (unsigned long long)comm->commHash, 
                  stream, 
                  recvcount * ncclTypeSize(datatype),
                  ncclTypeSize(datatype), 
                  op,
                  pid);

  eventAttrib_reducescatter.version = NVTX_VERSION;
  eventAttrib_reducescatter.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
  eventAttrib_reducescatter.messageType = NVTX_MESSAGE_TYPE_ASCII;
  eventAttrib_reducescatter.colorType = NVTX_COLOR_ARGB;
  eventAttrib_reducescatter.message.ascii = nvtxMsg_ReduceScatter;
  eventAttrib_reducescatter.color = colors[3];

  nvtxRangePushEx(&eventAttrib_reducescatter);
#endif

  NVTX3_FUNC_WITH_PARAMS(ReduceScatter, NcclNvtxParamsReduceScatter,
    NVTX3_PAYLOAD(comm ? comm->commHash : 0, recvcount * ncclTypeSize(datatype), op));

  struct ncclInfo info = { ncclFuncReduceScatter, "ReduceScatter",
    sendbuff, recvbuff, recvcount, datatype, op, 0, comm, stream, /* Args */
    REDUCESCATTER_CHUNKSTEPS, REDUCESCATTER_SLICESTEPS };
  
  ncclResult_t ret;
  ret = ncclEnqueueCheck(&info);

#if defined(ENABLE_API_NVTX)
  nvtxRangePop();
#endif

  return ret;
}

// TO DO
NCCL_API(ncclResult_t, ncclScatter, const void* sendbuff, void* recvbuff, size_t count,
    ncclDataType_t datatype, int root, ncclComm* comm, cudaStream_t stream);
ncclResult_t ncclScatter(const void* sendbuff, void* recvbuff, size_t count,
    ncclDataType_t datatype, int root, ncclComm* comm, cudaStream_t stream) {
  NVTX3_FUNC_WITH_PARAMS(Scatter, NcclNvtxParamsScatter,
    NVTX3_PAYLOAD(comm ? comm->commHash : 0, count * ncclTypeSize(datatype), root));

  struct ncclInfo info = { ncclFuncScatter, "Scatter",
    sendbuff, recvbuff, count, datatype, ncclSum, root, comm, stream, /* Args */
    SCATTER_CHUNKSTEPS, SCATTER_SLICESTEPS };
  return ncclEnqueueCheck(&info);
}

NCCL_API(ncclResult_t, ncclSend, const void* sendbuff, size_t count, ncclDataType_t datatype, int peer,
    ncclComm_t comm, cudaStream_t stream);
ncclResult_t ncclSend(const void* sendbuff, size_t count, ncclDataType_t datatype, int peer,
    ncclComm_t comm, cudaStream_t stream) {

  #if defined(ENABLE_API_NVTX)
  char nvtxMsg_Send[256];
  pid_t pid = getpid();
  snprintf(nvtxMsg_Send, sizeof(nvtxMsg_Send), 
                  "ncclSend(): commHash 0x%llx, stream %p, data_size %zu, type_size %d, receiver_rank %d, pid %d", 
                  (unsigned long long)comm->commHash, 
                  stream,
                  count * ncclTypeSize(datatype), 
                  ncclTypeSize(datatype), 
                  peer,
                  pid);

              eventAttrib_send.version = NVTX_VERSION;
              eventAttrib_send.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
              eventAttrib_send.messageType = NVTX_MESSAGE_TYPE_ASCII;
              eventAttrib_send.colorType = NVTX_COLOR_ARGB;
              eventAttrib_send.message.ascii = nvtxMsg_Send;
              eventAttrib_send.color = colors[5];

  nvtxRangePushEx(&eventAttrib_send);
#endif

  NVTX3_FUNC_WITH_PARAMS(Send, NcclNvtxParamsSendRecv,
    NVTX3_PAYLOAD(comm ? comm->commHash : 0, count * ncclTypeSize(datatype), peer));

  struct ncclInfo info = { ncclFuncSend, "Send",
    NULL, (void*)sendbuff, count, datatype, ncclSum, peer, comm, stream, /* Args */
    1, 1 };
 
  ncclResult_t ret;
  NCCLCHECK(ncclGroupStart());
  ret = ncclEnqueueCheck(&info);
  NCCLCHECK(ncclGroupEnd());

#if defined(ENABLE_API_NVTX)
  nvtxRangePop();
#endif

  return ret;
}

NCCL_API(ncclResult_t, ncclRecv, void* recvbuff, size_t count, ncclDataType_t datatype, int peer,
    ncclComm_t comm, cudaStream_t stream);
ncclResult_t ncclRecv(void* recvbuff, size_t count, ncclDataType_t datatype, int peer,
    ncclComm_t comm, cudaStream_t stream) {

#if defined(ENABLE_API_NVTX)
  char nvtxMsg_Recv[256];
  pid_t pid = getpid();
  snprintf(nvtxMsg_Recv, sizeof(nvtxMsg_Recv), 
                  "ncclRecv(): commHash 0x%llx, stream %p, data_size %zu, type_size %d, sender_rank %d, pid %d", 
                  (unsigned long long)comm->commHash, 
                  stream,
                  count * ncclTypeSize(datatype), 
                  ncclTypeSize(datatype), 
                  peer,
                  pid);

              eventAttrib_recv.version = NVTX_VERSION;
              eventAttrib_recv.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
              eventAttrib_recv.messageType = NVTX_MESSAGE_TYPE_ASCII;
              eventAttrib_recv.colorType = NVTX_COLOR_ARGB;
              eventAttrib_recv.message.ascii = nvtxMsg_Recv;
              eventAttrib_recv.color = colors[6];

  nvtxRangePushEx(&eventAttrib_recv);
#endif

  NVTX3_FUNC_WITH_PARAMS(Recv, NcclNvtxParamsSendRecv,
    NVTX3_PAYLOAD(comm ? comm->commHash : 0, count * ncclTypeSize(datatype), peer));

  struct ncclInfo info = { ncclFuncRecv, "Recv",
    NULL, recvbuff, count, datatype, ncclSum, peer, comm, stream, /* Args */
    1, 1 };
  ncclResult_t ret;
  NCCLCHECK(ncclGroupStart());
  ret = ncclEnqueueCheck(&info);
  NCCLCHECK(ncclGroupEnd());

#if defined(ENABLE_API_NVTX)
  nvtxRangePop();
#endif

  return ret;
}
