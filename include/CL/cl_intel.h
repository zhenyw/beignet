/* 
 * Copyright © 2012 Intel Corporation
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with this library. If not, see <http://www.gnu.org/licenses/>.
 *
 * Author: Benjamin Segovia <benjamin.segovia@intel.com>
 */

#ifndef __OPENCL_CL_INTEL_H
#define __OPENCL_CL_INTEL_H

#include "CL/cl.h"

#ifdef __cplusplus
extern "C" {
#endif

#define CL_MEM_PINNABLE (1 << 10)

/* Track allocations and report current number of unfreed allocations */
extern CL_API_ENTRY cl_int CL_API_CALL
clReportUnfreedIntel(void);

typedef CL_API_ENTRY cl_int (CL_API_CALL *clReportUnfreedIntel_fn)(void);

/* 1 to 1 mapping of drm_intel_bo_map */
extern CL_API_ENTRY void* CL_API_CALL
clMapBufferIntel(cl_mem, cl_int*);

typedef CL_API_ENTRY void* (CL_API_CALL *clMapBufferIntel_fn)(cl_mem, cl_int*);

/* 1 to 1 mapping of drm_intel_bo_unmap */
extern CL_API_ENTRY cl_int CL_API_CALL
clUnmapBufferIntel(cl_mem);

typedef CL_API_ENTRY cl_int (CL_API_CALL *clUnmapBufferIntel_fn)(cl_mem);

/* 1 to 1 mapping of drm_intel_gem_bo_map_gtt */
extern CL_API_ENTRY void* CL_API_CALL
clMapBufferGTTIntel(cl_mem, cl_int*);

typedef CL_API_ENTRY void* (CL_API_CALL *clMapBufferGTTIntel_fn)(cl_mem, cl_int*);

/* 1 to 1 mapping of drm_intel_gem_bo_unmap_gtt */
extern CL_API_ENTRY cl_int CL_API_CALL
clUnmapBufferGTTIntel(cl_mem);

typedef CL_API_ENTRY cl_int (CL_API_CALL *clUnmapBufferGTTIntel_fn)(cl_mem);

/* Pin /Unpin the buffer in GPU memory (must be root) */
extern CL_API_ENTRY cl_int CL_API_CALL
clPinBufferIntel(cl_mem);
extern CL_API_ENTRY cl_int CL_API_CALL
clUnpinBufferIntel(cl_mem);

typedef CL_API_ENTRY cl_int (CL_API_CALL *clPinBufferIntel_fn)(cl_mem);
typedef CL_API_ENTRY cl_int (CL_API_CALL *clUnpinBufferIntel_fn)(cl_mem);

/* Get the generation of the Gen device (used to load the proper binary) */
extern CL_API_ENTRY cl_int CL_API_CALL
clGetGenVersionIntel(cl_device_id device, cl_int *ver);

typedef CL_API_ENTRY cl_int (CL_API_CALL *clGetGenVersionIntel_fn)(
                             cl_device_id device,
                             cl_int *ver);

/* Create a program from a LLVM source file */
extern CL_API_ENTRY cl_program CL_API_CALL
clCreateProgramWithLLVMIntel(cl_context              /* context */,
                             cl_uint                 /* num_devices */,
                             const cl_device_id *    /* device_list */,
                             const char *            /* file */,
                             cl_int *                /* errcode_ret */);

typedef CL_API_ENTRY cl_program (CL_API_CALL *clCreateProgramWithLLVMIntel_fn)(
                                 cl_context              /* context */,
                                 cl_uint                 /* num_devices */,
                                 const cl_device_id *    /* device_list */,
                                 const char *            /* file */,
                                 cl_int *                /* errcode_ret */);

/* Create buffer from libva's buffer object */
extern CL_API_ENTRY cl_mem CL_API_CALL
clCreateBufferFromLibvaIntel(cl_context      /* context */,
                             unsigned int    /* bo_name */,
                             cl_int *        /* errcode_ret */);

typedef CL_API_ENTRY cl_mem (CL_API_CALL *clCreateBufferFromLibvaIntel_fn)(
                             cl_context     /* context */,
                             unsigned int   /* bo_name */,
                             cl_int *       /* errcode_ret */);

/* Create image from libva's buffer object */
typedef struct _cl_libva_image {
    unsigned int            bo_name;
    uint32_t                offset;
    uint32_t                width;
    uint32_t                height;
    cl_image_format         fmt;
    uint32_t                row_pitch;
    uint32_t                reserved[8];
} cl_libva_image;

extern CL_API_ENTRY cl_mem CL_API_CALL
clCreateImageFromLibvaIntel(cl_context               /* context */,
                            const cl_libva_image *   /* info */,
                            cl_int *                 /* errcode_ret */);

typedef CL_API_ENTRY cl_mem (CL_API_CALL *clCreateImageFromLibvaIntel_fn)(
                             cl_context             /* context */,
                             const cl_libva_image * /* info */,
                             cl_int *               /* errcode_ret */);

/* Create buffer from libva's buffer object */
extern CL_API_ENTRY cl_int CL_API_CALL
clGetMemObjectFdIntel(cl_context   /* context */,
                      cl_mem       /* Memory Obejct */,
                      int*         /* returned fd */);

typedef CL_API_ENTRY cl_int (CL_API_CALL *clGetMemObjectFdIntel_fn)(
                             cl_context   /* context */,
                             cl_mem       /* Memory Obejct */,
                             int*         /* returned fd */);

/* Intel performance query */

#define PERFQUERY_COUNTER_EVENT_INTEL                0x94F0
#define PERFQUERY_COUNTER_DURATION_NORM_INTEL        0x94F1
#define PERFQUERY_COUNTER_DURATION_RAW_INTEL         0x94F2
#define PERFQUERY_COUNTER_THROUGHPUT_INTEL           0x94F3
#define PERFQUERY_COUNTER_RAW_INTEL                  0x94F4
#define PERFQUERY_COUNTER_TIMESTAMP_INTEL            0x94F5

#define PERFQUERY_COUNTER_DATA_UINT32_INTEL          0x94F8
#define PERFQUERY_COUNTER_DATA_UINT64_INTEL          0x94F9
#define PERFQUERY_COUNTER_DATA_FLOAT_INTEL           0x94FA
#define PERFQUERY_COUNTER_DATA_DOUBLE_INTEL          0x94FB
#define PERFQUERY_COUNTER_DATA_BOOL32_INTEL          0x94FC
  
typedef struct perf_query_object *cl_perf_query_intel;
  
extern CL_API_ENTRY void CL_API_CALL
clGetFirstPerfQueryIdIntel(cl_context ctx, cl_uint *queryId);
typedef CL_API_ENTRY void (CL_API_CALL *clGetFirstPerfQueryIdIntel_fn)(cl_context ctx,
								       cl_uint *queryId);

extern CL_API_ENTRY void CL_API_CALL
clGetNextPerfQueryIdIntel(cl_context ctx, cl_uint queryId, cl_uint *nextQueryId);
typedef CL_API_ENTRY void (CL_API_CALL *clGetNextPerfQueryIdIntel_fn)(cl_context ctx,
								      cl_uint queryId,
								      cl_uint *nextQueryId);

extern CL_API_ENTRY void CL_API_CALL
clGetPerfQueryInfoIntel(cl_context ctx,
			cl_uint queryId,
			cl_uint queryNameLength, cl_char *queryName,
			cl_uint *dataSize, cl_uint *noCounters,
			cl_uint *noInstances);
typedef CL_API_ENTRY void (CL_API_CALL *clGetPerfQueryInfoIntel_fn)(cl_context ctx,
								    cl_uint queryId,
								    cl_uint queryNameLength,
								    cl_char *queryName,
								    cl_uint *dataSize,
								    cl_uint *noCounters,
								    cl_uint *noInstances);
  
extern CL_API_ENTRY void CL_API_CALL
clGetPerfCounterInfoIntel(cl_context ctx,
			  cl_uint queryId, cl_uint counterId,
			  cl_uint counterNameLength, cl_char *counterName,
			  cl_uint counterDescLength, cl_char *counterDesc,
			  cl_uint *counterOffset, cl_uint *counterDataSize,
			  cl_uint *counterTypeEnum, cl_uint *counterDataTypeEnum,
			  cl_ulong *rawCounterMaxValue);
typedef CL_API_ENTRY void
(CL_API_CALL *clGetPerfCounterInfoIntel_fn)(cl_context ctx,
					    cl_uint queryId,
					    cl_uint counterId,
					    cl_uint counterNameLength, cl_char *counterName,
					    cl_uint counterDescLength, cl_char *counterDesc,
					    cl_uint *counterOffset, cl_uint *counterDataSize,
					    cl_uint *counterTypeEnum, cl_uint *counterDataTypeEnum,
					    cl_ulong *rawCounterMaxValue);

extern CL_API_ENTRY cl_int CL_API_CALL
clCreatePerfQueryIntel(cl_context context, cl_uint queryId, cl_perf_query_intel *queryHandle);
typedef CL_API_ENTRY cl_int
(CL_API_CALL *clCreatePerfQueryIntel_fn)(cl_context context,
					 cl_uint queryId,
					 cl_perf_query_intel *queryHandle);

extern CL_API_ENTRY cl_int CL_API_CALL
clDeletePerfQueryIntel(cl_context context, cl_perf_query_intel queryHandle);
typedef CL_API_ENTRY cl_int
(CL_API_CALL *clDeletePerfQueryIntel_fn)(cl_context context,
					 cl_perf_query_intel queryHandle);

extern CL_API_ENTRY cl_int CL_API_CALL
clBeginPerfQueryIntel(cl_context context, cl_perf_query_intel queryHandle);
typedef CL_API_ENTRY cl_int (CL_API_CALL *clBeginPerfQueryIntel_fn)(cl_context context,
								    cl_perf_query_intel queryHandle);

extern CL_API_ENTRY cl_int CL_API_CALL
clEndPerfQueryIntel(cl_context context, cl_perf_query_intel queryHandle);
typedef CL_API_ENTRY cl_int (CL_API_CALL *clEndPerfQueryIntel_fn)(cl_context context,
								  cl_perf_query_intel queryHandle);

extern CL_API_ENTRY cl_int CL_API_CALL
clGetPerfQueryDataIntel(cl_context context,
			cl_perf_query_intel queryHandle,
			cl_uint flags, size_t dataSize, void *data,
			cl_uint *bytesWritten);
typedef CL_API_ENTRY cl_int
(CL_API_CALL *clGetPerfQueryDataIntel_fn)(cl_context context,
					  cl_perf_query_intel queryHandle,
					  cl_uint flags, size_t dataSize, void *data,
					  cl_uint *bytesWritten);

#ifdef __cplusplus
}
#endif

#endif /* __OPENCL_CL_INTEL_H */

