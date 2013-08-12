#include "utest_helper.hpp"

#define BUFFERSIZE  32*1024
void runtime_event(void)
{
  const size_t n = BUFFERSIZE;
  cl_int cpu_src[BUFFERSIZE];
  cl_event ev[3];
  cl_int status = 0;
  cl_int value = 34;

  // Setup kernel and buffers
  OCL_CREATE_KERNEL("compiler_event");
  OCL_CREATE_BUFFER(buf[0], 0, BUFFERSIZE*sizeof(int), NULL);

  for(cl_uint i=0; i<BUFFERSIZE; i++)
    cpu_src[i] = 3;

  OCL_CREATE_USER_EVENT(ev[0]);

  clEnqueueWriteBuffer(queue, buf[0], CL_TRUE, 0, BUFFERSIZE*sizeof(int), (void *)cpu_src, 1, &ev[0], &ev[1]);

  OCL_SET_ARG(0, sizeof(cl_mem), &buf[0]);
  OCL_SET_ARG(1, sizeof(int), &value);

  // Run the kernel
  globals[0] = n;
  locals[0] = 32;
  clEnqueueNDRangeKernel(queue, kernel, 1, NULL, globals, locals, 2, &ev[0], &ev[2]);

  for (cl_uint i = 0; i != sizeof(ev) / sizeof(cl_event); ++i) {
    clGetEventInfo(ev[i], CL_EVENT_COMMAND_EXECUTION_STATUS, sizeof(status), &status, NULL);
    OCL_ASSERT(status >= CL_SUBMITTED);
  }

  OCL_SET_USER_EVENT_STATUS(ev[0], CL_COMPLETE);

  clGetEventInfo(ev[0], CL_EVENT_COMMAND_EXECUTION_STATUS, sizeof(status), &status, NULL);
  OCL_ASSERT(status == CL_COMPLETE);

  OCL_FINISH();

  for (cl_uint i = 0; i != sizeof(ev) / sizeof(cl_event); ++i) {
    clGetEventInfo(ev[i], CL_EVENT_COMMAND_EXECUTION_STATUS, sizeof(status), &status, NULL);
    OCL_ASSERT(status <= CL_COMPLETE);
  }

  // Check results
  OCL_MAP_BUFFER(0);

  for (uint32_t i = 0; i < n; ++i) {
    OCL_ASSERT(((int*)buf_data[0])[i] == (int)value + 0x3);
  }
  OCL_UNMAP_BUFFER(0);

  for (cl_uint i = 0; i != sizeof(ev) / sizeof(cl_event); ++i) {
    clReleaseEvent(ev[i]);
  }
}

MAKE_UTEST_FROM_FUNCTION(runtime_event);
