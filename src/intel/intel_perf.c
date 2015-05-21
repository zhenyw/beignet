/*
 * Copyright 2015 Intel Corporation
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the
 * "Software"), to deal in the Software without restriction, including
 * without limitation the rights to use, copy, modify, merge, publish,
 * distribute, sub license, and/or sell copies of the Software, and to
 * permit persons to whom the Software is furnished to do so, subject to
 * the following conditions:
 *
 * The above copyright notice and this permission notice (including the
 * next paragraph) shall be included in all copies or substantial portions
 * of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
 * OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NON-INFRINGEMENT.
 * IN NO EVENT SHALL PRECISION INSIGHT AND/OR ITS SUPPLIERS BE LIABLE FOR
 * ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 *
 * Author:
 *     Zhenyu Wang <zhenyuw@linux.intel.com>
 */

#include <linux/perf_event.h>

#include <asm/unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/ioctl.h>
#include <stropts.h>

#include <limits.h>
#include <errno.h>
#include <stdint.h>
#include <stdbool.h>
#include <stdlib.h>
#include <stdarg.h>
#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <assert.h>
#include <inttypes.h>

#include "intel_driver.h"
#include "intel_perf.h"

#include "intel/intel_gpgpu.h"
#include "intel/intel_defines.h"
#include "intel/intel_batchbuffer.h"

#include "cl_context.h"
#include "cl_command_queue.h"
#include "cl_device_id.h"

#include "i915_drm.h"

#define DBG(fmt, args...) fprintf(stderr, fmt, ##args)

/* Describes how to read one OA counter which might be a raw counter read
 * directly from a counter snapshot or could be a higher level counter derived
 * from one or more raw counters.
 *
 * Raw counters will have set ->report_offset to the snapshot offset and have
 * an accumulator that can consider counter overflow according to the width of
 * that counter.
 *
 * Higher level counters can currently reference up to 3 other counters + use
 * ->config for anything. They don't need an accumulator.
 *
 * The data type that will be written to *value_out by the read function can
 * be determined by ->data_type
 */
struct perf_oa_counter
{
   struct perf_oa_counter *reference0;
   struct perf_oa_counter *reference1;
   struct perf_oa_counter *reference2;
   union {
      int report_offset;
      int config;
   };

   int accumulator_index;
   void (*accumulate)(struct perf_oa_counter *counter,
                      uint32_t *start,
                      uint32_t *end,
                      uint64_t *accumulator);
   unsigned int data_type;
   void (*read)(struct perf_oa_counter *counter,
                uint64_t *accumulated,
                void *value_out);
};

/* A counter that will be advertised and reported to applications */
struct perf_query_counter
{
   const char *name;
   const char *desc;
   unsigned int type;
   unsigned int data_type;
   uint64_t raw_max;
   size_t offset;
   size_t size;

   union {
      struct perf_oa_counter *oa_counter;
      uint32_t pipeline_stat_reg;
   };
};

struct perf_query_builder
{
   cl_context ctx;
   struct perf_query *query;
   size_t offset;
   int next_accumulator_index;

   int a_offset;
   int b_offset;
   int c_offset;

   struct perf_oa_counter *gpu_core_clock;
};

/**
 * i965 representation of a performance query object.
 *
 * NB: We want to keep this structure relatively lean considering that
 * applications may expect to allocate enough objects to be able to
 * query around all draw calls in a frame.
 */
struct perf_query_object
{
  const struct perf_query *query;

  /* Use own batch for perf bo */
  intel_batchbuffer_t *batch;
  
  struct {

    /**
     * BO containing OA counter snapshots at query Begin/End time.
     */
    drm_intel_bo *bo;
    int current_report_id;

    /**
     * We collect periodic counter snapshots via perf so we can account
     * for counter overflow and this is a pointer into the circular
     * perf buffer for collecting snapshots that lie within the begin-end
     * bounds of this query.
     */
    unsigned int perf_tail;

    /**
     * Storage the final accumulated OA counters.
     */
    uint64_t accumulator[MAX_RAW_OA_COUNTERS];

    /**
     * false while in the unresolved_elements list, and set to true when
     * the final, end MI_RPC snapshot has been accumulated.
     */
    bool results_accumulated;
    
  } oa;
};

/* Samples read from the perf circular buffer */
struct oa_perf_sample {
   struct perf_event_header header;
   uint32_t raw_size;
   uint8_t raw_data[];
};
#define MAX_OA_PERF_SAMPLE_SIZE (8 +   /* perf_event_header */       \
                                 4 +   /* raw_size */                \
                                 256 + /* raw OA counter snapshot */ \
                                 4)    /* alignment padding */

#define TAKEN(HEAD, TAIL, POT_SIZE)	(((HEAD) - (TAIL)) & (POT_SIZE - 1))

/* Note: this will equate to 0 when the buffer is exactly full... */
#define REMAINING(HEAD, TAIL, POT_SIZE) (POT_SIZE - TAKEN (HEAD, TAIL, POT_SIZE))

#if defined(__i386__)
#define rmb()           __asm__ volatile("lock; addl $0,0(%%esp)" ::: "memory")
#define mb()            __asm__ volatile("lock; addl $0,0(%%esp)" ::: "memory")
#endif

#if defined(__x86_64__)
#define rmb()           __asm__ volatile("lfence" ::: "memory")
#define mb()            __asm__ volatile("mfence" ::: "memory")
#endif

/* TODO: consider using <stdatomic.h> something like:
 *
 * #define rmb() atomic_thread_fence(memory_order_seq_consume)
 * #define mb() atomic_thread_fence(memory_order_seq_cst)
 */

/* Allow building for a more recent kernel than the system headers
 * correspond too... */
#ifndef PERF_EVENT_IOC_FLUSH
#include <linux/ioctl.h>
#define PERF_EVENT_IOC_FLUSH                 _IO ('$', 8)
#endif

#define SECOND_SNAPSHOT_OFFSET_IN_BYTES 2048

static inline size_t
pot_align(size_t base, int pot_alignment)
{
    return (base + pot_alignment - 1) & ~(pot_alignment - 1);
}

/******************************************************************************/
/**
 * Emit an MI_REPORT_PERF_COUNT command packet.
 *
 * This writes the current OA counter values to buffer.
 */
static void
emit_mi_report_perf_count(cl_context ctx,
                          struct perf_query_object *obj,
                          uint32_t offset_in_bytes,
                          uint32_t report_id)
{
  drm_intel_bo *bo = obj->oa.bo;

  assert(offset_in_bytes % 64 == 0);

  intel_batchbuffer_reset(obj->batch, 512);

  /* Reports apparently don't always get written unless we flush first. */
  /* XXX required? need to call pipe_control function in intel_gpgpu.c */
  //  intel_batchbuffer_emit_mi_flush(brw);

  BEGIN_BATCH(obj->batch, 3);
  OUT_BATCH(obj->batch, MI_REPORT_PERF_COUNT);
  OUT_RELOC(obj->batch, bo, I915_GEM_DOMAIN_RENDER, I915_GEM_DOMAIN_RENDER,
	    offset_in_bytes);
  OUT_BATCH(obj->batch, report_id);
  ADVANCE_BATCH(obj->batch);

  intel_batchbuffer_flush(obj->batch);
  
  /* XXX */
  /* Reports apparently don't always get written unless we flush after. */
  //intel_batchbuffer_emit_mi_flush(brw);
}

static unsigned int
read_perf_head(struct perf_event_mmap_page *mmap_page)
{
   unsigned int head = (*(volatile uint64_t *)&mmap_page->data_head);
   rmb();

   return head;
}

static void
write_perf_tail(struct perf_event_mmap_page *mmap_page,
                unsigned int tail)
{
   /* Make sure we've finished reading all the sample data we
    * we're consuming before updating the tail... */
   mb();
   mmap_page->data_tail = tail;
}

/* Update the real perf tail pointer according to the query tail that
 * is currently furthest behind...
 */
static void
update_perf_tail(cl_context ctx)
{
   unsigned int size = ctx->perfquery.perf_oa_buffer_size;
   unsigned int head = read_perf_head(ctx->perfquery.perf_oa_mmap_page);
   int straggler_taken = -1;
   unsigned int straggler_tail;
   int i;

   for (i = 0; i < ctx->perfquery.unresolved_elements; i++) {
      struct perf_query_object *obj = ctx->perfquery.unresolved[i];
      int taken;

      if (!obj->oa.bo)
         continue;

      taken = TAKEN(head, obj->oa.perf_tail, size);

      if (taken > straggler_taken) {
         straggler_taken = taken;
         straggler_tail = obj->oa.perf_tail;
      }
   }

   if (straggler_taken >= 0)
      write_perf_tail(ctx->perfquery.perf_oa_mmap_page, straggler_tail);
}

/**
 * Add a query to the global list of "unresolved queries."
 *
 * Queries are "unresolved" until all the counter snapshots have been
 * accumulated via accumulate_oa_snapshots() after the end MI_REPORT_PERF_COUNT
 * has landed in query->oa.bo.
 */
static void
add_to_unresolved_query_list(cl_context ctx,
                             struct perf_query_object *obj)
{
   if (ctx->perfquery.unresolved_elements >=
       ctx->perfquery.unresolved_array_size) {
      ctx->perfquery.unresolved_array_size *= 1.5;
      ctx->perfquery.unresolved = realloc(ctx->perfquery.unresolved,
                                          sizeof(struct perf_query_object *) *
					  ctx->perfquery.unresolved_array_size);
   }

   ctx->perfquery.unresolved[ctx->perfquery.unresolved_elements++] = obj;

   if (obj->oa.bo)
      update_perf_tail(ctx);
}

/**
 * Remove a query from the global list of "unresolved queries." once
 * the end MI_RPC OA counter snapshot has been accumulated, or when
 * discarding unwanted query results.
 */
static void
drop_from_unresolved_query_list(cl_context ctx,
                                struct perf_query_object *obj)
{
  int i;
  
  for (i = 0; i < ctx->perfquery.unresolved_elements; i++) {
    if (ctx->perfquery.unresolved[i] == obj) {
      int last_elt = --ctx->perfquery.unresolved_elements;
      
      if (i == last_elt)
	ctx->perfquery.unresolved[i] = NULL;
      else
	ctx->perfquery.unresolved[i] = ctx->perfquery.unresolved[last_elt];
      
      break;
    }
  }

  if (obj->oa.bo)
    update_perf_tail(ctx);
}

static uint64_t
read_report_timestamp(cl_context ctx, uint32_t *report)
{
   return ctx->perfquery.read_oa_report_timestamp(report);
}

/**
 * Given pointers to starting and ending OA snapshots, add the deltas for each
 * counter to the results.
 */
static void
add_deltas(cl_context ctx,
           struct perf_query_object *obj,
           uint32_t *start, uint32_t *end)
{
   const struct perf_query *query = obj->query;
   int i;
   
#if 0
   fprintf(stderr, "Accumulating delta:\n");
   fprintf(stderr, "> Start timestamp = %" PRIu64 "\n", read_report_timestamp(ctx, start));
   fprintf(stderr, "> End timestamp = %" PRIu64 "\n", read_report_timestamp(ctx, end));
#endif

   for (i = 0; i < query->n_oa_counters; i++) {
      struct perf_oa_counter *oa_counter = &query->oa_counters[i];
      //uint64_t pre_accumulate;

      if (!oa_counter->accumulate)
         continue;

      //pre_accumulate = query->oa.accumulator[counter->id];
      oa_counter->accumulate(oa_counter,
                             start, end,
                             obj->oa.accumulator);
#if 0
      fprintf(stderr, "> Updated %s from %" PRIu64 " to %" PRIu64 "\n",
              counter->name, pre_accumulate,
              query->oa.accumulator[counter->id]);
#endif
   }
}

/* Handle restarting ioctl if interupted... */
static int
perf_ioctl(int fd, unsigned long request, void *arg)
{
   int ret;

   do {
      ret = ioctl(fd, request, arg);
   } while (ret == -1 && (errno == EINTR || errno == EAGAIN));
   return ret;
}

static bool
inc_n_oa_users(cl_context ctx)
{
   if (ctx->perfquery.n_oa_users == 0 &&
       perf_ioctl(ctx->perfquery.perf_oa_event_fd,
                  PERF_EVENT_IOC_ENABLE, 0) < 0)
   {
      return false;
   }
   ++ctx->perfquery.n_oa_users;

   return true;
}

static void
dec_n_oa_users(cl_context ctx)
{
   /* Disabling the i915_oa event will effectively disable the OA
    * counters.  Note it's important to be sure there are no outstanding
    * MI_RPC commands at this point since they could stall the CS
    * indefinitely once OACONTROL is disabled.
    */
   --ctx->perfquery.n_oa_users;
   if (ctx->perfquery.n_oa_users == 0 &&
       perf_ioctl(ctx->perfquery.perf_oa_event_fd,
                  PERF_EVENT_IOC_DISABLE, 0) < 0)
   {
      DBG("WARNING: Error disabling i915_oa perf event: %m\n");
   }
}

/**
 * Accumulate OA counter results from a series of snapshots.
 *
 * N.B. We write snapshots for the beginning and end of a query into
 * query->oa.bo as well as collect periodic snapshots from the Linux
 * perf interface.
 *
 * These periodic snapshots help to ensure we handle counter overflow
 * correctly by being frequent enough to ensure we don't miss multiple
 * wrap overflows of a counter between snapshots.
 */
static void
accumulate_oa_snapshots(cl_context ctx,
                        struct perf_query_object *obj)
{
   uint32_t *query_buffer;
   uint8_t *data = ctx->perfquery.perf_oa_mmap_base + ctx->perfquery.page_size;
   const unsigned int size = ctx->perfquery.perf_oa_buffer_size;
   const uint64_t mask = size - 1;
   uint64_t head;
   uint64_t tail;
   uint32_t *start;
   uint64_t start_timestamp;
   uint32_t *last;
   uint32_t *end;
   uint64_t end_timestamp;
   uint8_t scratch[MAX_OA_PERF_SAMPLE_SIZE];

   if (perf_ioctl(ctx->perfquery.perf_oa_event_fd,
                  PERF_EVENT_IOC_FLUSH, 0) < 0)
      DBG("Failed to flush outstanding perf events: %m\n");

   drm_intel_bo_map(obj->oa.bo, false);
   query_buffer = obj->oa.bo->virtual;

   start = last = query_buffer;
   end = query_buffer + (SECOND_SNAPSHOT_OFFSET_IN_BYTES / sizeof(uint32_t));

#warning "TODO: find a way to report OA errors from the kernel"
   /* XXX: Is there anything we can do to handle this gracefully/
    * report the error to the application? */
   if (start[0] != obj->oa.current_report_id)
      DBG("Spurious start report id=%"PRIu32"\n", start[0]);
   if (end[0] != (obj->oa.current_report_id + 1))
      DBG("Spurious end report id=%"PRIu32"\n", start[0]);

   start_timestamp = read_report_timestamp(ctx, start);
   end_timestamp = read_report_timestamp(ctx, end);

   head = read_perf_head(ctx->perfquery.perf_oa_mmap_page);
   tail = obj->oa.perf_tail;

   //fprintf(stderr, "Handle event mask = 0x%" PRIx64
   //        " head=%" PRIu64 " tail=%" PRIu64 "\n", mask, head, tail);

   while (TAKEN(head, tail, size)) {
      const struct perf_event_header *header =
         (const struct perf_event_header *)(data + (tail & mask));

      if (header->size == 0) {
         DBG("Spurious header size == 0\n");
         /* XXX: How should we handle this instead of exiting() */
#warning "FIXME: avoid exit(1) in error condition"
         exit(1);
      }

      if (header->size > (head - tail)) {
         DBG("Spurious header size would overshoot head\n");
         /* XXX: How should we handle this instead of exiting() */
         exit(1);
      }

      //fprintf(stderr, "header = %p tail=%" PRIu64 " size=%d\n",
      //        header, tail, header->size);

      if ((const uint8_t *)header + header->size > data + size) {
         int before;

         if (header->size > MAX_OA_PERF_SAMPLE_SIZE) {
            DBG("Skipping spurious sample larger than expected\n");
            tail += header->size;
            continue;
         }

         before = data + size - (const uint8_t *)header;

         memcpy(scratch, header, before);
         memcpy(scratch + before, data, header->size - before);

         header = (struct perf_event_header *)scratch;
         //fprintf(stderr, "DEBUG: split\n");
         //exit(1);
      }

      switch (header->type) {
         case PERF_RECORD_LOST: {
            struct {
               struct perf_event_header header;
               uint64_t id;
               uint64_t n_lost;
            } *lost = (void *)header;
            DBG("i915_oa: Lost %" PRIu64 " events\n", lost->n_lost);
            break;
         }

         case PERF_RECORD_THROTTLE:
            DBG("i915_oa: Sampling has been throttled\n");
            break;

         case PERF_RECORD_UNTHROTTLE:
            DBG("i915_oa: Sampling has been unthrottled\n");
            break;

         case PERF_RECORD_SAMPLE: {
            struct oa_perf_sample *perf_sample = (struct oa_perf_sample *)header;
            uint32_t *report = (uint32_t *)perf_sample->raw_data;
            uint64_t timestamp = read_report_timestamp(ctx, report);

            if (timestamp >= end_timestamp)
               goto end;

            if (timestamp > start_timestamp) {
               add_deltas(ctx, obj, last, report);
               last = report;
            }

            break;
         }

         default:
            DBG("i915_oa: Spurious header type = %d\n", header->type);
      }

      //fprintf(stderr, "Tail += %d\n", header->size);

      tail += header->size;
   }

end:

   add_deltas(ctx, obj, last, end);

   DBG("Marking %p resolved - results gathered\n", obj);

   drm_intel_bo_unmap(obj->oa.bo);
   obj->oa.results_accumulated = true;
   drop_from_unresolved_query_list(ctx, obj);
   dec_n_oa_users(ctx);
}

/******************************************************************************/

static uint64_t
read_file_uint64 (const char *file)
{
   char buf[32];
   int fd, n;

   fd = open(file, 0);
   if (fd < 0)
      return 0;
   n = read(fd, buf, sizeof (buf) - 1);
   close(fd);
   if (n < 0)
      return 0;

   buf[n] = '\0';
   return strtoull(buf, 0, 0);
}

static uint64_t
lookup_i915_oa_id (void)
{
   return read_file_uint64("/sys/bus/event_source/devices/i915_oa/type");
}

static long
perf_event_open (struct perf_event_attr *hw_event,
                 pid_t pid,
                 int cpu,
                 int group_fd,
                 unsigned long flags)
{
   return syscall(__NR_perf_event_open, hw_event, pid, cpu, group_fd, flags);
}

static bool
open_i915_oa_event(cl_context ctx,
                   int metrics_set,
                   int report_format,
                   int period_exponent,
                   int drm_fd,
                   uint32_t ctx_id)
{
   struct perf_event_attr attr;
   drm_i915_oa_attr_t oa_attr;
   int event_fd;
   void *mmap_base;

   memset(&attr, 0, sizeof(attr));
   attr.size = sizeof(attr);
   attr.type = lookup_i915_oa_id();

   attr.sample_type = PERF_SAMPLE_RAW;
   attr.disabled = 1; /* initially off */
   attr.sample_period = 1;

   memset(&oa_attr, 0, sizeof(oa_attr));
   oa_attr.size = sizeof(oa_attr);

   oa_attr.format = report_format;
   oa_attr.metrics_set = metrics_set;
   oa_attr.timer_exponent = period_exponent;

   oa_attr.single_context = true;
   oa_attr.ctx_id = ctx_id;
   oa_attr.drm_fd = drm_fd;

   attr.config = (uint64_t)&oa_attr;

   event_fd = perf_event_open(&attr,
                              -1, /* pid */
                              0, /* cpu */
                              -1, /* group fd */
                              PERF_FLAG_FD_CLOEXEC); /* flags */
   if (event_fd == -1) {
      DBG("Error opening i915_oa perf event: %m\n");
      return false;
   }

   /* NB: A read-write mapping ensures the kernel will stop writing data when
    * the buffer is full, and will report samples as lost. */
   mmap_base = mmap(NULL,
                    ctx->perfquery.perf_oa_buffer_size + ctx->perfquery.page_size,
                    PROT_READ | PROT_WRITE, MAP_SHARED, event_fd, 0);
   if (mmap_base == MAP_FAILED) {
      DBG("Error mapping circular buffer, %m\n");
      close (event_fd);
      return false;
   }

   ctx->perfquery.perf_oa_event_fd = event_fd;
   ctx->perfquery.perf_oa_mmap_base = mmap_base;
   ctx->perfquery.perf_oa_mmap_page = mmap_base;

   ctx->perfquery.perf_oa_metrics_set = metrics_set;
   ctx->perfquery.perf_oa_format = report_format;

   return true;
}

static bool
begin_perf_query(cl_context ctx,
		 struct perf_query_object *obj)
{
   const struct perf_query *query = obj->query;
   intel_driver_t *driver = (intel_driver_t *)ctx->drv;

   /* If the OA counters aren't already on, enable them. */
   if (ctx->perfquery.perf_oa_event_fd == -1) {
     uint32_t ctx_id = drm_intel_gem_context_get_context_id(driver->ctx);
     int period_exponent;
     
     /* The timestamp for HSW+ increments every 80ns
      *
      * The period_exponent gives a sampling period as follows:
      *   sample_period = 80ns * 2^(period_exponent + 1)
      *
      * The overflow period for Haswell can be calculated as:
      *
      * 2^32 / (n_eus * max_gen_freq * 2)
      * (E.g. 40 EUs @ 1GHz = ~53ms)
      *
      * We currently sample every 42 milliseconds...
      */
     period_exponent = 18;
     
     if (!open_i915_oa_event(ctx,
			     query->oa_metrics_set,
			     query->oa_format,
			     period_exponent,
			     driver->fd,
			     ctx_id))
       return false;
   } else {
     /* Opening an i915_oa event fd implies exclusive access to
      * the OA unit which will generate counter reports for a
      * specific counter set/profile with a specific layout/format
      * so we can't begin any OA based queries that require a
      * different profile or format unless we get an opportunity
      * to close the event fd and open a new one...
      */
     if (ctx->perfquery.perf_oa_metrics_set != query->oa_metrics_set ||
	 ctx->perfquery.perf_oa_format != query->oa_format)
       {
	 return false;
       }
   }

   if (!inc_n_oa_users(ctx)) {
     DBG("WARNING: Error enabling i915_oa perf event: %m\n");
     return false;
   }

   if (obj->oa.bo) {
     drm_intel_bo_unreference(obj->oa.bo);
     obj->oa.bo = NULL;
   }

   obj->oa.bo =
     drm_intel_bo_alloc(driver->bufmgr, "perf. query OA bo", 4096, 64);
#ifdef DEBUG
   /* Pre-filling the BO helps debug whether writes landed. */
   drm_intel_bo_map(obj->oa.bo, true);
   memset((char *) obj->oa.bo->virtual, 0x80, 4096);
   drm_intel_bo_unmap(obj->oa.bo);
#endif
   
   obj->oa.current_report_id = ctx->perfquery.next_query_start_report_id;
   ctx->perfquery.next_query_start_report_id += 2;

   /* Take a starting OA counter snapshot. */
   emit_mi_report_perf_count(ctx, obj, 0,
			     obj->oa.current_report_id);
   ++ctx->perfquery.n_active_oa_queries;

   /* Each unresolved query maintains a separate tail pointer into the
    * circular perf sample buffer. The real tail pointer in
    * perfquery.perf_oa_mmap_page.data_tail will correspond to the query
    * tail that is furthest behind.
    */
   obj->oa.perf_tail = read_perf_head(ctx->perfquery.perf_oa_mmap_page);
   
   memset(obj->oa.accumulator, 0, sizeof(obj->oa.accumulator));
   obj->oa.results_accumulated = false;
   
   add_to_unresolved_query_list(ctx, obj);

   return true;
}

static void
end_perf_query(cl_context ctx,
	       struct perf_query_object *obj)
{
  /* Take an ending OA counter snapshot. */
  emit_mi_report_perf_count(ctx, obj,
			    SECOND_SNAPSHOT_OFFSET_IN_BYTES,
			    obj->oa.current_report_id + 1);
  --ctx->perfquery.n_active_oa_queries;

  /* NB: even though the query has now ended, it can't be resolved
   * until the end MI_REPORT_PERF_COUNT snapshot has been written
   * to query->oa.bo */
}

static void
wait_perf_query(cl_context ctx,
		struct perf_query_object *obj)
{
  drm_intel_bo *bo = NULL;

  bo = obj->oa.bo;
  if (bo == NULL)
    return;

   /* If the current batch references our results bo then we need to
    * flush first... */
   if (drm_intel_bo_references(obj->batch->buffer, bo))
      intel_batchbuffer_flush(obj->batch);

   if (drm_intel_bo_busy(bo))
         DBG("Stalling GPU waiting for a performance query object.\n");

   drm_intel_bo_wait_rendering(bo);
}

/**
 * Is a performance query result available?
 */
static bool
is_perf_query_ready(cl_context ctx,
		    struct perf_query_object *obj)
{
   return (obj->oa.results_accumulated ||
	   (obj->oa.bo &&
	    !drm_intel_bo_references(obj->batch->buffer, obj->oa.bo) && 
	    !drm_intel_bo_busy(obj->oa.bo)));
}


/******************************************************************************/

/* Type safe wrappers for reading OA counter values */

static uint64_t
read_uint64_oa_counter(struct perf_oa_counter *counter, uint64_t *accumulated)
{
   uint64_t value;

   assert(counter->data_type == PERFQUERY_COUNTER_DATA_UINT64_INTEL);

   counter->read(counter, accumulated, &value);

   return value;
}

static float
read_float_oa_counter(struct perf_oa_counter *counter, uint64_t *accumulated)
{
   float value;

   assert(counter->data_type == PERFQUERY_COUNTER_DATA_FLOAT_INTEL);

   counter->read(counter, accumulated, &value);

   return value;
}

/******************************************************************************/

/*
 * OA counter normalisation support...
 */

static void
read_accumulated_oa_counter_cb(struct perf_oa_counter *counter,
                               uint64_t *accumulator,
                               void *value)
{
   *((uint64_t *)value) = accumulator[counter->accumulator_index];
}

static void
accumulate_uint32_cb(struct perf_oa_counter *counter,
                     uint32_t *report0,
                     uint32_t *report1,
                     uint64_t *accumulator)
{
   accumulator[counter->accumulator_index] +=
      (uint32_t)(report1[counter->report_offset] -
                 report0[counter->report_offset]);
}

#if 0
/* XXX: we should factor this out for now, but notably BDW has 40bit counters... */
static void
accumulate_uint40_cb(struct perf_oa_counter *counter,
                     uint32_t *report0,
                     uint32_t *report1,
                     uint64_t *accumulator)
{
   uint32_t value0 = report0[counter->report_offset];
   uint32_t value1 = report1[counter->report_offset];
   uint64_t delta;

   if (value0 > value1)
      delta = (1ULL << 40) + value1 - value0;
   else
      delta = value1 - value0;

   accumulator[counter->accumulator_index] += delta;
}
#endif

static struct perf_oa_counter *
add_raw_oa_counter(struct perf_query_builder *builder, int report_offset)
{
   struct perf_oa_counter *counter =
      &builder->query->oa_counters[builder->query->n_oa_counters++];

   counter->report_offset = report_offset;
   counter->accumulator_index = builder->next_accumulator_index++;
   counter->accumulate = accumulate_uint32_cb;
   counter->read = read_accumulated_oa_counter_cb;
   counter->data_type = PERFQUERY_COUNTER_DATA_UINT64_INTEL;

   return counter;
}

static uint64_t
hsw_read_report_timestamp(uint32_t *report)
{
   /* The least significant timestamp bit represents 80ns on Haswell */
   return ((uint64_t)report[1]) * 80;
}

static void
accumulate_hsw_elapsed_cb(struct perf_oa_counter *counter,
                          uint32_t *report0,
                          uint32_t *report1,
                          uint64_t *accumulator)
{
   uint64_t timestamp0 = hsw_read_report_timestamp(report0);
   uint64_t timestamp1 = hsw_read_report_timestamp(report1);

   accumulator[counter->accumulator_index] += (timestamp1 - timestamp0);
}

static struct perf_oa_counter *
add_hsw_elapsed_oa_counter(struct perf_query_builder *builder)
{
   struct perf_oa_counter *counter =
      &builder->query->oa_counters[builder->query->n_oa_counters++];

   counter->accumulator_index = builder->next_accumulator_index++;
   counter->accumulate = accumulate_hsw_elapsed_cb;
   counter->read = read_accumulated_oa_counter_cb;
   counter->data_type = PERFQUERY_COUNTER_DATA_UINT64_INTEL;

   return counter;
}

static void
read_frequency_cb(struct perf_oa_counter *counter,
                  uint64_t *accumulated,
                  void *value) /* uint64 */
{
   uint64_t clk_delta = read_uint64_oa_counter(counter->reference0, accumulated);
   uint64_t time_delta = read_uint64_oa_counter(counter->reference1, accumulated);
   uint64_t *ret = value;

   if (!clk_delta) {
      *ret = 0;
      return;
   }

   *ret = (clk_delta * 1000) / time_delta;
}

static struct perf_oa_counter *
add_avg_frequency_oa_counter(struct perf_query_builder *builder,
                             struct perf_oa_counter *timestamp)
{
   struct perf_oa_counter *counter =
      &builder->query->oa_counters[builder->query->n_oa_counters++];

   assert(timestamp->data_type == PERFQUERY_COUNTER_DATA_UINT64_INTEL);

   counter->reference0 = builder->gpu_core_clock;
   counter->reference1 = timestamp;
   counter->read = read_frequency_cb;
   counter->data_type = PERFQUERY_COUNTER_DATA_UINT64_INTEL;

   return counter;
}

static void
read_oa_counter_normalized_by_gpu_duration_cb(struct perf_oa_counter *counter,
                                              uint64_t *accumulated,
                                              void *value) /* float */
{
   uint64_t delta = read_uint64_oa_counter(counter->reference0, accumulated);
   uint64_t clk_delta = read_uint64_oa_counter(counter->reference1, accumulated);
   float *ret = value;

   if (!clk_delta) {
      *ret = 0;
      return;
   }

   *ret = ((double)delta * 100.0) / (double)clk_delta;
}

static struct perf_oa_counter *
add_oa_counter_normalised_by_gpu_duration(struct perf_query_builder *builder,
                                          struct perf_oa_counter *raw)
{
   struct perf_oa_counter *counter =
      &builder->query->oa_counters[builder->query->n_oa_counters++];

   counter->reference0 = raw;
   counter->reference1 = builder->gpu_core_clock;
   counter->read = read_oa_counter_normalized_by_gpu_duration_cb;
   counter->data_type = PERFQUERY_COUNTER_DATA_FLOAT_INTEL;

   return counter;
}

static void
read_hsw_samplers_busy_duration_cb(struct perf_oa_counter *counter,
                                   uint64_t *accumulated,
                                   void *value) /* float */
{
   uint64_t sampler0_busy = read_uint64_oa_counter(counter->reference0, accumulated);
   uint64_t sampler1_busy = read_uint64_oa_counter(counter->reference1, accumulated);
   uint64_t clk_delta = read_uint64_oa_counter(counter->reference2, accumulated);
   float *ret = value;

   if (!clk_delta) {
      *ret = 0;
      return;
   }

   *ret = ((double)(sampler0_busy + sampler1_busy) * 100.0) / ((double)clk_delta * 2.0);
}

static struct perf_oa_counter *
add_hsw_samplers_busy_duration_oa_counter(struct perf_query_builder *builder,
                                          struct perf_oa_counter *sampler0_busy_raw,
                                          struct perf_oa_counter *sampler1_busy_raw)
{
   struct perf_oa_counter *counter =
      &builder->query->oa_counters[builder->query->n_oa_counters++];

   counter->reference0 = sampler0_busy_raw;
   counter->reference1 = sampler1_busy_raw;
   counter->reference2 = builder->gpu_core_clock;
   counter->read = read_hsw_samplers_busy_duration_cb;
   counter->data_type = PERFQUERY_COUNTER_DATA_FLOAT_INTEL;

   return counter;
}

static void
read_hsw_slice_extrapolated_cb(struct perf_oa_counter *counter,
                               uint64_t *accumulated,
                               void *value) /* float */
{
   uint64_t counter0 = read_uint64_oa_counter(counter->reference0, accumulated);
   uint64_t counter1 = read_uint64_oa_counter(counter->reference1, accumulated);
   int eu_count = counter->config;
   uint64_t *ret = value;

   *ret = (counter0 + counter1) * eu_count;
}

static struct perf_oa_counter *
add_hsw_slice_extrapolated_oa_counter(struct perf_query_builder *builder,
                                      struct perf_oa_counter *counter0,
                                      struct perf_oa_counter *counter1)
{
   struct perf_oa_counter *counter =
      &builder->query->oa_counters[builder->query->n_oa_counters++];

   counter->reference0 = counter0;
   counter->reference1 = counter1;
   counter->config = builder->ctx->perfquery.eu_count;
   counter->read = read_hsw_slice_extrapolated_cb;
   counter->data_type = PERFQUERY_COUNTER_DATA_UINT64_INTEL;

   return counter;
}

static void
read_oa_counter_normalized_by_eu_duration_cb(struct perf_oa_counter *counter,
                                             uint64_t *accumulated,
                                             void *value) /* float */
{
   uint64_t delta = read_uint64_oa_counter(counter->reference0, accumulated);
   uint64_t clk_delta = read_uint64_oa_counter(counter->reference1, accumulated);
   float *ret = value;

   if (!clk_delta) {
      *ret = 0;
      return;
   }

   delta /= counter->config; /* EU count */

   *ret = (double)delta * 100.0 / (double)clk_delta;
}

static struct perf_oa_counter *
add_oa_counter_normalised_by_eu_duration(struct perf_query_builder *builder,
                                         struct perf_oa_counter *raw)
{
   struct perf_oa_counter *counter =
      &builder->query->oa_counters[builder->query->n_oa_counters++];

   counter->reference0 = raw;
   counter->reference1 = builder->gpu_core_clock;
   counter->config = builder->ctx->perfquery.eu_count;
   counter->read = read_oa_counter_normalized_by_eu_duration_cb;
   counter->data_type = PERFQUERY_COUNTER_DATA_FLOAT_INTEL;

   return counter;
}

static void
read_av_thread_cycles_counter_cb(struct perf_oa_counter *counter,
                                 uint64_t *accumulated,
                                 void *value) /* uint64 */
{
   uint64_t delta = read_uint64_oa_counter(counter->reference0, accumulated);
   uint64_t spawned = read_uint64_oa_counter(counter->reference1, accumulated);
   uint64_t *ret = value;

   if (!spawned) {
      *ret = 0;
      return;
   }

   *ret = delta / spawned;
}

static struct perf_oa_counter *
add_average_thread_cycles_oa_counter(struct perf_query_builder *builder,
                                     struct perf_oa_counter *raw,
                                     struct perf_oa_counter *denominator)
{
   struct perf_oa_counter *counter =
      &builder->query->oa_counters[builder->query->n_oa_counters++];

   counter->reference0 = raw;
   counter->reference1 = denominator;
   counter->read = read_av_thread_cycles_counter_cb;
   counter->data_type = PERFQUERY_COUNTER_DATA_UINT64_INTEL;

   return counter;
}

static void
read_scaled_uint64_counter_cb(struct perf_oa_counter *counter,
                              uint64_t *accumulated,
                              void *value) /* uint64 */
{
   uint64_t delta = read_uint64_oa_counter(counter->reference0, accumulated);
   uint64_t scale = counter->config;
   uint64_t *ret = value;

   *ret = delta * scale;
}

static struct perf_oa_counter *
add_scaled_uint64_oa_counter(struct perf_query_builder *builder,
                             struct perf_oa_counter *input,
                             int scale)
{
   struct perf_oa_counter *counter =
      &builder->query->oa_counters[builder->query->n_oa_counters++];

   counter->reference0 = input;
   counter->config = scale;
   counter->read = read_scaled_uint64_counter_cb;
   counter->data_type = PERFQUERY_COUNTER_DATA_UINT64_INTEL;

   return counter;
}

static void
read_max_of_float_counters_cb(struct perf_oa_counter *counter,
                              uint64_t *accumulated,
                              void *value) /* float */
{
   float counter0 = read_float_oa_counter(counter->reference0, accumulated);
   float counter1 = read_float_oa_counter(counter->reference1, accumulated);
   float *ret = value;

   *ret = counter0 >= counter1 ? counter0 : counter1;
}


static struct perf_oa_counter *
add_max_of_float_oa_counters(struct perf_query_builder *builder,
                             struct perf_oa_counter *counter0,
                             struct perf_oa_counter *counter1)
{
   struct perf_oa_counter *counter =
      &builder->query->oa_counters[builder->query->n_oa_counters++];

   counter->reference0 = counter0;
   counter->reference1 = counter1;
   counter->read = read_max_of_float_counters_cb;
   counter->data_type = PERFQUERY_COUNTER_DATA_FLOAT_INTEL;

   return counter;
}

static void
report_uint64_oa_counter_as_raw_uint64(struct perf_query_builder *builder,
                                       const char *name,
                                       const char *desc,
                                       struct perf_oa_counter *oa_counter)
{
   struct perf_query_counter *counter =
      &builder->query->counters[builder->query->n_counters++];

   counter->oa_counter = oa_counter;
   counter->name = name;
   counter->desc = desc;
   counter->type = PERFQUERY_COUNTER_RAW_INTEL;
   counter->data_type = PERFQUERY_COUNTER_DATA_UINT64_INTEL;
   counter->raw_max = 0; /* undefined range */
   counter->offset = pot_align(builder->offset, 8);
   counter->size = sizeof(uint64_t);

   builder->offset = counter->offset + counter->size;
}

static void
report_uint64_oa_counter_as_uint64_event(struct perf_query_builder *builder,
                                         const char *name,
                                         const char *desc,
                                         struct perf_oa_counter *oa_counter)
{
   struct perf_query_counter *counter =
      &builder->query->counters[builder->query->n_counters++];

   counter->oa_counter = oa_counter;
   counter->name = name;
   counter->desc = desc;
   counter->type = PERFQUERY_COUNTER_EVENT_INTEL;
   counter->data_type = PERFQUERY_COUNTER_DATA_UINT64_INTEL;
   counter->offset = pot_align(builder->offset, 8);
   counter->size = sizeof(uint64_t);

   builder->offset = counter->offset + counter->size;
}

static void
report_float_oa_counter_as_percentage_duration(struct perf_query_builder *builder,
                                               const char *name,
                                               const char *desc,
                                               struct perf_oa_counter *oa_counter)
{
   struct perf_query_counter *counter =
      &builder->query->counters[builder->query->n_counters++];

   counter->oa_counter = oa_counter;
   counter->name = name;
   counter->desc = desc;
   counter->type = PERFQUERY_COUNTER_DURATION_RAW_INTEL;
   counter->data_type = PERFQUERY_COUNTER_DATA_FLOAT_INTEL;
   counter->raw_max = 100;
   counter->offset = pot_align(builder->offset, 4);
   counter->size = sizeof(float);

   builder->offset = counter->offset + counter->size;
}

static void
report_uint64_oa_counter_as_throughput(struct perf_query_builder *builder,
                                       const char *name,
                                       const char *desc,
                                       struct perf_oa_counter *oa_counter)
{
   struct perf_query_counter *counter =
      &builder->query->counters[builder->query->n_counters++];

   counter->oa_counter = oa_counter;
   counter->name = name;
   counter->desc = desc;
   counter->type = PERFQUERY_COUNTER_THROUGHPUT_INTEL;
   counter->data_type = PERFQUERY_COUNTER_DATA_UINT64_INTEL;
   counter->offset = pot_align(builder->offset, 8);
   counter->size = sizeof(uint64_t);

   builder->offset = counter->offset + counter->size;
}

static void
report_uint64_oa_counter_as_duration(struct perf_query_builder *builder,
                                     const char *name,
                                     const char *desc,
                                     struct perf_oa_counter *oa_counter)
{
   struct perf_query_counter *counter =
      &builder->query->counters[builder->query->n_counters++];

   counter->oa_counter = oa_counter;
   counter->name = name;
   counter->desc = desc;
   counter->type = PERFQUERY_COUNTER_DURATION_RAW_INTEL;
   counter->data_type = PERFQUERY_COUNTER_DATA_UINT64_INTEL;
   counter->raw_max = 0;
   counter->offset = pot_align(builder->offset, 8);
   counter->size = sizeof(uint64_t);

   builder->offset = counter->offset + counter->size;
}

static void
add_aggregate_counters(struct perf_query_builder *builder)
{
   struct perf_oa_counter *raw;
   struct perf_oa_counter *active, *stall, *n_threads;
   struct perf_oa_counter *c;
   int a_offset = builder->a_offset;
   int aggregate_active_counter = a_offset + 17; /* aggregate active */
   int aggregate_stall_counter = a_offset + 18; /* aggregate stall */
   int n_threads_counter = a_offset + 20;

   raw = add_raw_oa_counter(builder, a_offset + 41);
   c = add_oa_counter_normalised_by_gpu_duration(builder, raw);
   report_float_oa_counter_as_percentage_duration(builder,
                                                  "GPU Busy",
                                                  "The percentage of time in which the GPU has being processing GPU commands.",
                                                  c);

   raw = add_raw_oa_counter(builder, a_offset); /* aggregate EU active */
   c = add_oa_counter_normalised_by_eu_duration(builder, raw);
   report_float_oa_counter_as_percentage_duration(builder,
                                                   "EU Active",
                                                   "The percentage of time in which the Execution Units were actively processing.",
                                                   c);

   raw = add_raw_oa_counter(builder, a_offset + 1); /* aggregate EU stall */
   c = add_oa_counter_normalised_by_eu_duration(builder, raw);
   report_float_oa_counter_as_percentage_duration(builder,
                                                   "EU Stall",
                                                   "The percentage of time in which the Execution Units were stalled.",
                                                   c);


   active = add_raw_oa_counter(builder, aggregate_active_counter);
   c = add_oa_counter_normalised_by_eu_duration(builder, active);
   report_float_oa_counter_as_percentage_duration(builder,
						  "CS EU Active",
						  "The percentage of time in which compute shader were "
						  "processed actively on the EUs.",
						  c);

   stall = add_raw_oa_counter(builder, aggregate_stall_counter);
   c = add_oa_counter_normalised_by_eu_duration(builder, stall);
   report_float_oa_counter_as_percentage_duration(builder,
						  "CS EU Stall",
						  "The percentage of time in which compute shader were "
						  "stalled on the EUs.",
						  c);


   n_threads = add_raw_oa_counter(builder, n_threads_counter);
   c = add_average_thread_cycles_oa_counter(builder, active, n_threads);
   report_uint64_oa_counter_as_raw_uint64(builder,
					  "CS AVG Active per Thread",
					  "The average number of cycles per hardware "
					  "thread run in which compute shader were processed actively "
					  "on the EUs.",
					  c);

   c = add_average_thread_cycles_oa_counter(builder, stall, n_threads);
   report_uint64_oa_counter_as_raw_uint64(builder,
					  "CS AVG Stall per Thread",
					  "The average number of cycles per hardware "
					  "thread run in which compute shader were stalled "
					  "on the EUs.",
					  c);
   

   #if 0
   raw = add_raw_oa_counter(builder, a_offset + 32); /* hiz fast z passing */
   raw = add_raw_oa_counter(builder, a_offset + 33); /* hiz fast z failing */

   raw = add_raw_oa_counter(builder, a_offset + 42); /* vs bottleneck */
   raw = add_raw_oa_counter(builder, a_offset + 43); /* gs bottleneck */
   #endif
}

static void
hsw_add_compute_counters(struct perf_query_builder *builder)
{
    struct perf_oa_counter *raw0;
    struct perf_oa_counter *raw1;
    struct perf_oa_counter *typed_read;
    struct perf_oa_counter *typed_write;
    struct perf_oa_counter *typed_atomics;
    struct perf_oa_counter *untyped_read;
    struct perf_oa_counter *untyped_write;
    struct perf_oa_counter *slm_read;
    struct perf_oa_counter *slm_write;

    raw0 = add_raw_oa_counter(builder, 0xd0>>2);
    raw0 = add_scaled_uint64_oa_counter(builder, raw0, 64);

    raw1 = add_raw_oa_counter(builder, 0xd4>>2);
    raw1 = add_scaled_uint64_oa_counter(builder, raw0, 64);

    typed_read = add_hsw_slice_extrapolated_oa_counter(builder, raw0, raw1);
    report_uint64_oa_counter_as_raw_uint64(builder,
					   "TYPED_BYTES_READ",
					   "TYPED_BYTES_READ",
					   typed_read);

    raw0 = add_raw_oa_counter(builder, 0xd8>>2);
    raw0 = add_scaled_uint64_oa_counter(builder, raw0, 64);

    raw1 = add_raw_oa_counter(builder, 0xdc>>2);
    raw1 = add_scaled_uint64_oa_counter(builder, raw1, 64);

    typed_write = add_hsw_slice_extrapolated_oa_counter(builder, raw0, raw1);
    report_uint64_oa_counter_as_raw_uint64(builder,
					   "TYPED_BYTES_WRITTEN",
					   "TYPED_BYTES_WRITTEN",
					   typed_write);

    raw0 = add_raw_oa_counter(builder, 0xc0>>2);
    raw0 = add_scaled_uint64_oa_counter(builder, raw0, 64);

    raw1 = add_raw_oa_counter(builder, 0xc4>>2);
    raw1 = add_scaled_uint64_oa_counter(builder, raw1, 64);

    untyped_read = add_hsw_slice_extrapolated_oa_counter(builder, raw0, raw1);
    report_uint64_oa_counter_as_raw_uint64(builder,
					   "UNTYPED_BYTES_READ",
					   "UNTYPED_BYTES_READ",
					   untyped_read);

    raw0 = add_raw_oa_counter(builder, 0xc8>>2);
    raw0 = add_scaled_uint64_oa_counter(builder, raw0, 64);

    raw1 = add_raw_oa_counter(builder, 0xcc>>2);
    raw1 = add_scaled_uint64_oa_counter(builder, raw1, 64);

    untyped_write = add_hsw_slice_extrapolated_oa_counter(builder, raw0, raw1);
    report_uint64_oa_counter_as_raw_uint64(builder,
					   "UNTYPED_BYTES_WRITTEN",
					   "UNTYPED_BYTES_WRITTEN",
					   untyped_write);

    raw0 = add_raw_oa_counter(builder, 0xf8>>2);
    raw0 = add_scaled_uint64_oa_counter(builder, raw0, 64);

    raw1 = add_raw_oa_counter(builder, 0xfc>>2);
    raw1 = add_scaled_uint64_oa_counter(builder, raw1, 64);

    slm_read = add_hsw_slice_extrapolated_oa_counter(builder, raw0, raw1);
    report_uint64_oa_counter_as_raw_uint64(builder,
					   "SLM_BYTES_READ",
					   "SLM_BYTES_READ",
					   slm_read);

    raw0 = add_raw_oa_counter(builder, 0xf0>>2);
    raw0 = add_scaled_uint64_oa_counter(builder, raw0, 64);

    raw1 = add_raw_oa_counter(builder, 0xf4>>2);
    raw1 = add_scaled_uint64_oa_counter(builder, raw1, 64);

    slm_write = add_hsw_slice_extrapolated_oa_counter(builder, raw0, raw1);
    report_uint64_oa_counter_as_raw_uint64(builder,
					   "SLM_BYTES_WRITTEN",
					   "SLM_BYTES_WRITTEN",
					   slm_write);

    raw0 = add_raw_oa_counter(builder, 0xe0>>2);
    raw1 = add_raw_oa_counter(builder, 0xe4>>2);
    typed_atomics = add_hsw_slice_extrapolated_oa_counter(builder, raw0, raw1);
    report_uint64_oa_counter_as_raw_uint64(builder,
					   "TYPED_ATOMICS",
					   "TYPED_ATOMICS",
					   typed_atomics);
}


static void
hsw_add_compute_basic_oa_counter_query(cl_context ctx)
{
    struct perf_query_builder builder;
    struct perf_query *query = &ctx->perfquery.queries[ctx->perfquery.n_queries++];
    int a_offset;
    int b_offset;
    int c_offset;
    struct perf_oa_counter *elapsed;
    struct perf_oa_counter *c;
    struct perf_query_counter *last;

    query->name = "Gen7 Compute Basic Observability Architecture Counters";
    query->counters = malloc(sizeof(struct perf_query_counter) *
			     MAX_PERF_QUERY_COUNTERS);
    query->n_counters = 0;
    query->oa_counters = malloc(sizeof(struct perf_oa_counter) *
				MAX_OA_QUERY_COUNTERS);
    query->n_oa_counters = 0;
    query->oa_metrics_set = I915_OA_METRICS_SET_COMPUTE;
    query->oa_format = I915_OA_FORMAT_A45_B8_C8_HSW;
    //    query->perf_raw_size = 256; /* XXX */

    builder.ctx = ctx;
    builder.query = query;
    builder.offset = 0;
    builder.next_accumulator_index = 0;

    builder.a_offset = a_offset = 3;
    builder.b_offset = b_offset = a_offset + 45;
    builder.c_offset = c_offset = b_offset + 8;

    /* Can be referenced by other counters... */
    builder.gpu_core_clock = add_raw_oa_counter(&builder, c_offset + 2);

    elapsed = add_hsw_elapsed_oa_counter(&builder);
    report_uint64_oa_counter_as_duration(&builder,
					 "GPU Time Elapsed",
					 "Time elapsed on the GPU during the measurement.",
					 elapsed);

    c = add_avg_frequency_oa_counter(&builder, elapsed);
    report_uint64_oa_counter_as_uint64_event(&builder,
					     "AVG GPU Core Frequency",
					     "Average GPU Core Frequency in the measurement.",
					     c);

    add_aggregate_counters(&builder);

    hsw_add_compute_counters(&builder);

    assert(query->n_counters < MAX_PERF_QUERY_COUNTERS);
    assert(query->n_oa_counters < MAX_OA_QUERY_COUNTERS);
    
    last = &query->counters[query->n_counters - 1];
    query->data_size = last->offset + last->size;
}

static int
get_oa_counter_data(cl_context ctx,
                    struct perf_query_object *obj,
                    size_t data_size,
                    uint8_t *data)
{
   const struct perf_query *query = obj->query;
   int n_counters = query->n_counters;
   int written = 0, i;

   if (!obj->oa.results_accumulated) {
      accumulate_oa_snapshots(ctx, obj);
      assert(obj->oa.results_accumulated);
   }

   for (i = 0; i < n_counters; i++) {
      const struct perf_query_counter *counter = &query->counters[i];

      if (counter->size) {
         counter->oa_counter->read(counter->oa_counter, obj->oa.accumulator,
                                   data + counter->offset);
         written = counter->offset + counter->size;
      }
   }

   return written;
}

/**
 * Get the performance query result.
 */
static int
get_perf_query_data(cl_context ctx,
		    struct perf_query_object *obj,
		    cl_uint flags,
		    size_t data_size,
		    cl_uint *data,
		    cl_uint *bytes_written)
{
   int written = 0;

   if (!is_perf_query_ready(ctx, obj)) {
     if (flags == PERFQUERY_FLUSH_INTEL) {
       if (drm_intel_bo_references(obj->batch->buffer, obj->oa.bo))
	 intel_batchbuffer_flush(obj->batch);
     }

     /* Currently just ensure our perf query bo finish
      * as we use our own batch for submit. Future might
      * refactor with normal queue batch to sync with kernel
      * finish point e.g clFinish
      */
     wait_perf_query(ctx, obj);
   }

   written = get_oa_counter_data(ctx, obj, data_size, (uint8_t *)data);
   
   if (bytes_written)
      *bytes_written = written;

   return 0;
}

static void
close_perf(cl_context ctx)
{
   if (ctx->perfquery.perf_oa_event_fd != -1) {
      if (ctx->perfquery.perf_oa_mmap_base) {
         size_t mapping_len =
            ctx->perfquery.perf_oa_buffer_size + ctx->perfquery.page_size;

         munmap(ctx->perfquery.perf_oa_mmap_base, mapping_len);
         ctx->perfquery.perf_oa_mmap_base = NULL;
      }

      close(ctx->perfquery.perf_oa_event_fd);
      ctx->perfquery.perf_oa_event_fd = -1;
   }
}

void
intel_perf_query_first(cl_context ctx, cl_uint *queryId)
{
  *queryId = 0;
}

void
intel_perf_query_next(cl_context ctx, cl_uint queryId, cl_uint *nextId)
{
  if (queryId < ctx->perfquery.n_queries - 1)
    *nextId = queryId + 1;
  else
    *nextId = queryId;
}

void
intel_perf_query_info(cl_context ctx,
		      cl_uint queryId,
		      cl_char **queryName,
		      cl_uint *dataSize,
		      cl_uint *noCounters,
		      cl_uint *noInstances)
{
  struct perf_query *query = &ctx->perfquery.queries[queryId];

  *queryName = query->name;
  *dataSize = query->data_size;
  *noCounters = query->n_counters;
  *noInstances = 1; /* current initial allows 1 instance */
}

void
intel_perf_counter_info(cl_context ctx,
			cl_uint queryId, cl_uint counterId,
			cl_char **counterName,
			cl_char **counterDesc,
			cl_uint *counterOffset, cl_uint *counterDataSize,
			cl_uint *counterTypeEnum, cl_uint *counterDataTypeEnum,
			cl_ulong *rawCounterMaxValue)
{
  struct perf_query *query = &ctx->perfquery.queries[queryId];
  struct perf_query_counter *counter = &query->counters[counterId];

  *counterName = counter->name;
  *counterDesc = counter->desc;
  *counterOffset = counter->offset;
  *counterDataSize = counter->size;
  *counterTypeEnum = counter->type;
  *counterDataTypeEnum = counter->data_type;
  *rawCounterMaxValue = counter->raw_max;
}

cl_int
intel_perf_query_create(cl_context context, cl_uint queryId, cl_perf_query_intel *queryHandle)
{
  struct perf_query *query = &context->perfquery.queries[queryId];
  struct perf_query_object *obj =
    calloc(1, sizeof(struct perf_query_object));

  if (!obj)
    return -1;

  obj->query = query;

  obj->batch = intel_batchbuffer_new((intel_driver_t *)context->drv);
  if (!obj->batch) {
    fprintf(stderr, "failed to create perf batch\n");
    free(obj);
    return -1;
  }

  context->perfquery.n_query_instances++;

  *queryHandle = obj;

  return 0;
}

cl_int
intel_perf_query_delete(cl_context context, cl_perf_query_intel queryHandle)
{
  struct perf_query_object *obj = (struct perf_query_object *)queryHandle;
  
  if (obj->oa.bo) {
    if (!obj->oa.results_accumulated) {
      drop_from_unresolved_query_list(context, obj);
      dec_n_oa_users(context);
    }

    drm_intel_bo_unreference(obj->oa.bo);
    obj->oa.bo = NULL;
  }

  obj->oa.results_accumulated = false;

  intel_batchbuffer_delete(obj->batch);
  
  free(obj);

  if (--context->perfquery.n_query_instances == 0)
    close_perf(context);

  return 0;
}

cl_int
intel_perf_query_begin(cl_context context, cl_perf_query_intel queryHandle)
{
  bool ret;
  struct perf_query_object *obj = (struct perf_query_object *)queryHandle;
  
  ret = begin_perf_query(context, obj);
  if (!ret)
    return -1;

  return 0;
}

cl_int
intel_perf_query_end(cl_context context, cl_perf_query_intel queryHandle)
{
  struct perf_query_object *obj = (struct perf_query_object *)queryHandle;

  end_perf_query(context, obj);
  return 0;
}

cl_int
intel_perf_query_get_data(cl_context context,
			  cl_perf_query_intel queryHandle,
			  cl_uint flags, size_t dataSize, void *data,
			  cl_uint *bytesWritten)
{
  struct perf_query_object *obj = (struct perf_query_object *)queryHandle;

  return get_perf_query_data(context, obj, flags,
			     dataSize, data,
			     bytesWritten);
}


void
intel_perf_query_init(cl_context context)
{
  intel_driver_t *drv = (intel_driver_t *)context->drv;

  if (!IS_HASWELL(drv->device_id)) {
    fprintf(stderr, "Perf query only supports on HSW now.\n");
    context->perfquery.enable = false;
    return;
  }

  /* XXX test kernel for i915 PMU driver available */
  if (lookup_i915_oa_id() == 0) {
    fprintf(stderr, "Kernel has no i915 PMU driver.\n");
    context->perfquery.enable = false;
    return;
  }
  
  context->perfquery.eu_count = context->device->max_compute_unit;

  context->perfquery.read_oa_report_timestamp = hsw_read_report_timestamp;

  /* initialize intel query structs depends on device */
  hsw_add_compute_basic_oa_counter_query(context);

  context->perfquery.unresolved = calloc(2, sizeof(struct perf_query_object *));
  context->perfquery.unresolved_elements = 0;
  context->perfquery.unresolved_array_size = 2;

  context->perfquery.page_size = sysconf(_SC_PAGE_SIZE);

  context->perfquery.perf_oa_event_fd = -1;
  context->perfquery.perf_oa_buffer_size = 1024 * 1024; /* NB: must be power of two */

  context->perfquery.next_query_start_report_id = 1000;

  context->perfquery.enable = true;
}

void
intel_perf_query_destroy(cl_context context)
{
  int i;
  struct perf_query *query;

  for (i = 0; i < context->perfquery.n_queries; i++) {
    query = &context->perfquery.queries[i];
    free(query->counters);
    free(query->oa_counters);
  }
  free(context->perfquery.unresolved);
}
