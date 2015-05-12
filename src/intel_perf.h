#ifndef _INTEL_PERF_H
#define _INTEL_PERF_H

#include "CL/cl.h"
#include "CL/cl_intel.h"

void intel_perf_query_first(cl_context, cl_uint *queryId);
void intel_perf_query_next(cl_context, cl_uint queryId, cl_uint *nextId);
void intel_perf_query_info(cl_context, cl_uint queryId,
			   cl_char **queryName,
			   cl_uint *dataSize, cl_uint *noCounters, cl_uint *noInstances);
void intel_perf_counter_info(cl_context, cl_uint queryId, cl_uint counterId,
			     cl_char **counterName,
			     cl_char **counterDesc,
			     cl_uint *counterOffset, cl_uint *counterDataSize,
			     cl_uint *counterTypeEnum, cl_uint *counterDataTypeEnum,
			     cl_ulong *rawCounterMaxValue);
cl_int intel_perf_query_create(cl_context context, cl_uint queryId,
			       cl_perf_query_intel *queryHandle);
cl_int intel_perf_query_delete(cl_context context, cl_perf_query_intel queryHandle);
cl_int intel_perf_query_begin(cl_context context, cl_perf_query_intel queryHandle);
cl_int intel_perf_query_end(cl_context context, cl_perf_query_intel queryHandle);
cl_int intel_perf_query_get_data(cl_context context,
				 cl_perf_query_intel queryHandle,
				 cl_uint flags, size_t dataSize, void *data,
				 cl_uint *bytesWritten);

void intel_perf_query_init(cl_context context);
void intel_perf_query_destroy(cl_context context);

#endif

