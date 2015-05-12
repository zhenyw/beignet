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

#ifndef __CL_CONTEXT_H__
#define __CL_CONTEXT_H__

#include "CL/cl.h"
#include "cl_internals.h"
#include "cl_driver.h"
#include "cl_khr_icd.h"

#include <stdint.h>
#include <stdbool.h>
#include <pthread.h>

/* DRI device created at create context */
struct intel_driver;

enum _cl_gl_context_type {
  CL_GL_NOSHARE,
  CL_GL_EGL_DISPLAY,
  CL_GL_GLX_DISPLAY,
  CL_GL_WGL_HDC,
  CL_GL_CGL_SHAREGROUP
};

enum _cl_internal_ker_type {
  CL_INTERNAL_KERNEL_MIN = 0,
  CL_ENQUEUE_COPY_BUFFER_ALIGN4 = 0,
  CL_ENQUEUE_COPY_BUFFER_ALIGN16,
  CL_ENQUEUE_COPY_BUFFER_UNALIGN_SAME_OFFSET,
  CL_ENQUEUE_COPY_BUFFER_UNALIGN_DST_OFFSET,
  CL_ENQUEUE_COPY_BUFFER_UNALIGN_SRC_OFFSET,
  CL_ENQUEUE_COPY_BUFFER_RECT,
  CL_ENQUEUE_COPY_BUFFER_RECT_ALIGN4,
  CL_ENQUEUE_COPY_IMAGE_1D_TO_1D,             //copy image 1d to image 1d
  CL_ENQUEUE_COPY_IMAGE_2D_TO_2D,             //copy image 2d to image 2d
  CL_ENQUEUE_COPY_IMAGE_3D_TO_2D,             //copy image 3d to image 2d
  CL_ENQUEUE_COPY_IMAGE_2D_TO_3D,             //copy image 2d to image 3d
  CL_ENQUEUE_COPY_IMAGE_3D_TO_3D,             //copy image 3d to image 3d
  CL_ENQUEUE_COPY_IMAGE_2D_TO_2D_ARRAY,       //copy image 2d to image 2d array
  CL_ENQUEUE_COPY_IMAGE_1D_ARRAY_TO_1D_ARRAY, //copy image 1d array to image 1d array
  CL_ENQUEUE_COPY_IMAGE_2D_ARRAY_TO_2D_ARRAY, //copy image 2d array to image 2d array
  CL_ENQUEUE_COPY_IMAGE_2D_ARRAY_TO_2D,       //copy image 2d array to image 2d
  CL_ENQUEUE_COPY_IMAGE_2D_ARRAY_TO_3D,       //copy image 2d array to image 3d
  CL_ENQUEUE_COPY_IMAGE_3D_TO_2D_ARRAY,       //copy image 3d to image 2d array
  CL_ENQUEUE_COPY_IMAGE_2D_TO_BUFFER,   //copy image 2d to buffer
  CL_ENQUEUE_COPY_IMAGE_2D_TO_BUFFER_ALIGN16,
  CL_ENQUEUE_COPY_IMAGE_3D_TO_BUFFER,   //copy image 3d tobuffer
  CL_ENQUEUE_COPY_BUFFER_TO_IMAGE_2D,   //copy buffer to image 2d
  CL_ENQUEUE_COPY_BUFFER_TO_IMAGE_2D_ALIGN16,
  CL_ENQUEUE_COPY_BUFFER_TO_IMAGE_3D,   //copy buffer to image 3d
  CL_ENQUEUE_FILL_BUFFER_UNALIGN,      //fill buffer with 1 aligne pattern, pattern size=1
  CL_ENQUEUE_FILL_BUFFER_ALIGN2,       //fill buffer with 2 aligne pattern, pattern size=2
  CL_ENQUEUE_FILL_BUFFER_ALIGN4,       //fill buffer with 4 aligne pattern, pattern size=4
  CL_ENQUEUE_FILL_BUFFER_ALIGN8_8,     //fill buffer with 8 aligne pattern, pattern size=8
  CL_ENQUEUE_FILL_BUFFER_ALIGN8_16,    //fill buffer with 16 aligne pattern, pattern size=16
  CL_ENQUEUE_FILL_BUFFER_ALIGN8_32,    //fill buffer with 16 aligne pattern, pattern size=32
  CL_ENQUEUE_FILL_BUFFER_ALIGN8_64,    //fill buffer with 16 aligne pattern, pattern size=64
  CL_ENQUEUE_FILL_BUFFER_ALIGN128,     //fill buffer with 128 aligne pattern, pattern size=128
  CL_ENQUEUE_FILL_IMAGE_1D,             //fill image 1d
  CL_ENQUEUE_FILL_IMAGE_1D_ARRAY,       //fill image 1d array
  CL_ENQUEUE_FILL_IMAGE_2D,             //fill image 2d
  CL_ENQUEUE_FILL_IMAGE_2D_ARRAY,       //fill image 2d array
  CL_ENQUEUE_FILL_IMAGE_3D,             //fill image 3d
  CL_INTERNAL_KERNEL_MAX
};

struct _cl_context_prop {
  cl_context_properties platform_id;
  enum _cl_gl_context_type gl_type;
  cl_context_properties gl_context;
  union {
    cl_context_properties egl_display;
    cl_context_properties glx_display;
    cl_context_properties wgl_hdc;
    cl_context_properties cgl_sharegroup;
  };
};

struct perf_query_counter;
struct perf_oa_counter;

struct perf_query
{
   const char *name;
   struct perf_query_counter *counters;
   int n_counters;
   size_t data_size;

   /* OA specific */
   int oa_metrics_set;
   int oa_format;
   struct perf_oa_counter *oa_counters;
   int n_oa_counters;
};

#define MAX_PERF_QUERIES 2
#define MAX_PERF_QUERY_COUNTERS 150
#define MAX_OA_QUERY_COUNTERS 100
#define MAX_RAW_OA_COUNTERS 62

#define IS_EGL_CONTEXT(ctx)  (ctx->props.gl_type == CL_GL_EGL_DISPLAY)
#define EGL_DISP(ctx)   (EGLDisplay)(ctx->props.egl_display)
#define EGL_CTX(ctx)    (EGLContext)(ctx->props.gl_context)
/* Encapsulate the whole device */
struct _cl_context {
  DEFINE_ICD(dispatch)
  uint64_t magic;                   /* To identify it as a context */
  volatile int ref_n;               /* We reference count this object */
  cl_driver drv;                    /* Handles HW or simulator */
  cl_device_id device;              /* All information about the GPU device */
  cl_command_queue queues;          /* All command queues currently allocated */
  cl_program programs;              /* All programs currently allocated */
  cl_mem buffers;                   /* All memory object currently allocated */
  cl_sampler samplers;              /* All sampler object currently allocated */
  cl_event   events;                /* All event object currently allocated */
  pthread_mutex_t queue_lock;       /* To allocate and deallocate queues */
  pthread_mutex_t program_lock;     /* To allocate and deallocate programs */
  pthread_mutex_t buffer_lock;      /* To allocate and deallocate buffers */
  pthread_mutex_t sampler_lock;     /* To allocate and deallocate samplers */
  pthread_mutex_t event_lock;       /* To allocate and deallocate events */
  cl_program internal_prgs[CL_INTERNAL_KERNEL_MAX];
                                    /* All programs internal used, for example clEnqueuexxx api use */
  cl_kernel  internel_kernels[CL_INTERNAL_KERNEL_MAX];
                                    /* All kernels  for clenqueuexxx api, for example clEnqueuexxx api use */
  cl_program built_in_prgs;  /*all built-in kernels belongs to this program only*/
  cl_kernel  built_in_kernels[CL_INTERNAL_KERNEL_MAX];
  uint32_t ver;                     /* Gen version */
  struct _cl_context_prop props;
  cl_context_properties * prop_user; /* a copy of user passed context properties when create context */
  cl_uint                 prop_len;  /* count of the properties */
  void (CL_CALLBACK *pfn_notify)(const char *, const void *, size_t, void *);
                                     /* User's callback when error occur in context */
  void *user_data;                   /* A pointer to user supplied data */

  struct {
    struct perf_query queries[MAX_PERF_QUERIES];
    int n_queries;
    bool enable;

    /* A common OA counter that we want to read directly in several places */
    uint64_t (*read_oa_report_timestamp)(uint32_t *report);

    /* Needed to normalize counters aggregated across all EUs */
    int eu_count;

    /* The i915_oa perf event we open to setup + enable the OA counters */
    int perf_oa_event_fd;

    /* An i915_oa perf event fd gives exclusive access to the OA unit that
     * will report counter snapshots for a specific counter set/profile in a
     * specific layout/format so we can only start OA queries that are
     * compatible with the currently open fd... */
    int perf_oa_metrics_set;
    int perf_oa_format;

    /* The mmaped circular buffer for collecting samples from perf */
    uint8_t *perf_oa_mmap_base;
    size_t perf_oa_buffer_size;
    struct perf_event_mmap_page *perf_oa_mmap_page;

    /* The system's page size */
    unsigned int page_size;

    /* TODO: generalize and split these into an array indexed by the
     * query type... */
    int n_active_oa_queries;

    /* The number of queries depending on running OA counters which
     * extends beyond brw_end_perf_query() since we need to wait until
     * the last MI_RPC command has been written. */
    int n_oa_users;

    /* We also get the gpu to write an ID for snapshots corresponding
     * to the beginning and end of a query, but for simplicity these
     * IDs use a separate namespace. */
    int next_query_start_report_id;

    /**
     * An array of queries whose results haven't yet been assembled based on
     * the data in buffer objects.
     *
     * These may be active, or have already ended.  However, the results
     * have not been requested.
     */
    struct perf_query_object **unresolved;
    int unresolved_elements;
    int unresolved_array_size;

    /* The total number of query objects so we can relinquish
     * our exclusive access to perf if the application deletes
     * all of its objects. (NB: We only disable perf while
     * there are no active queries) */
    int n_query_instances;
  } perfquery;
};

/* Implement OpenCL function */
extern cl_context cl_create_context(const cl_context_properties*,
                                    cl_uint,
                                    const cl_device_id*,
                                    void (CL_CALLBACK * pfn_notify) (const char*, const void*, size_t, void*),
                                    void *,
                                    cl_int*);

/* Allocate and initialize a context */
extern cl_context cl_context_new(struct _cl_context_prop *);

/* Destroy and deallocate a context */
extern void cl_context_delete(cl_context);

/* Increment the context reference counter */
extern void cl_context_add_ref(cl_context);

/* Create the command queue from the given context and device */
extern cl_command_queue cl_context_create_queue(cl_context,
                                                cl_device_id,
                                                cl_command_queue_properties,
                                                cl_int*);

/* Enqueue a ND Range kernel */
extern cl_int cl_context_ND_kernel(cl_context,
                                   cl_command_queue,
                                   cl_kernel,
                                   cl_uint,
                                   const size_t*,
                                   const size_t*,
                                   const size_t*);

/* Used for allocation */
extern cl_buffer_mgr cl_context_get_bufmgr(cl_context ctx);

/* Get the internal used kernel */
extern cl_kernel cl_context_get_static_kernel(cl_context ctx, cl_int index, const char *str_kernel, const char * str_option);

/* Get the internal used kernel from binary*/
extern cl_kernel cl_context_get_static_kernel_from_bin(cl_context ctx, cl_int index,
                  const char * str_kernel, size_t size, const char * str_option);

#endif /* __CL_CONTEXT_H__ */

