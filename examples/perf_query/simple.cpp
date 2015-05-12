//
// File:       hello.c
//
// Abstract:   A simple "Hello World" compute example showing basic usage of OpenCL which
//             calculates the mathematical square (X[i] = pow(X[i],2)) for a buffer of
//             floating point values.
//             
//

////////////////////////////////////////////////////////////////////////////////

#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <CL/opencl.h>

#include <CL/cl_intel.h>

////////////////////////////////////////////////////////////////////////////////

// Use a static data size for simplicity
//
#define DATA_SIZE (1024)

////////////////////////////////////////////////////////////////////////////////

// Simple compute kernel which computes the square of an input array 
//
const char *KernelSource = "\n" \
"__kernel void square(                                                       \n" \
"   __global float* input,                                              \n" \
"   __global float* output,                                             \n" \
"   const unsigned int count)                                           \n" \
"{                                                                      \n" \
"   int i = get_global_id(0);                                           \n" \
"   if(i < count)                                                       \n" \
"       output[i] = input[i] * input[i];                                \n" \
"}                                                                      \n" \
"\n";

////////////////////////////////////////////////////////////////////////////////

clGetFirstPerfQueryIdIntel_fn cl_get_first_perf_query_id;
clGetNextPerfQueryIdIntel_fn cl_get_next_perf_query_id;
clGetPerfQueryInfoIntel_fn cl_get_perf_query_info;
clGetPerfCounterInfoIntel_fn cl_get_perf_counter_info;
clCreatePerfQueryIntel_fn cl_create_perf_query;
clDeletePerfQueryIntel_fn cl_delete_perf_query;
clBeginPerfQueryIntel_fn cl_begin_perf_query;
clEndPerfQueryIntel_fn cl_end_perf_query;
clGetPerfQueryDataIntel_fn cl_get_perf_query_data;

struct perf_counter {
  cl_uint id;
  cl_char name[60];
  cl_char desc[120];
  cl_uint offset;
  cl_uint size;
  cl_uint type;
  cl_uint data_type;
  cl_ulong raw_max;
};

int main(int argc, char** argv)
{
    int err;                            // error code returned from api calls
      
    float data[DATA_SIZE];              // original data set given to device
    float results[DATA_SIZE];           // results returned from device
    unsigned int correct;               // number of correct results returned

    size_t global;                      // global domain size for our calculation
    size_t local;                       // local domain size for our calculation

    cl_platform_id platform_id;
    cl_device_id device_id;             // compute device id 
    cl_context context;                 // compute context
    cl_command_queue commands;          // compute command queue
    cl_program program;                 // compute program
    cl_kernel kernel;                   // compute kernel
    
    cl_mem input;                       // device memory used for the input array
    cl_mem output;                      // device memory used for the output array
    
    // Fill our data set with random float values
    //
    int i = 0;
    unsigned int count = DATA_SIZE;
    for(i = 0; i < count; i++)
        data[i] = rand() / (float)RAND_MAX;
    
    // Connect to a compute device
    //
    cl_uint n_platform;
    clGetPlatformIDs(1, &platform_id, &n_platform);
    printf("Num of platform: %d\n", n_platform);
    
    int gpu = 1;
    err = clGetDeviceIDs(platform_id, gpu ? CL_DEVICE_TYPE_GPU : CL_DEVICE_TYPE_CPU, 1, &device_id, NULL);
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to create a device group!\n");
        return EXIT_FAILURE;
    }
  
    // Create a compute context 
    //
    context = clCreateContext(0, 1, &device_id, NULL, NULL, &err);
    if (!context)
    {
        printf("Error: Failed to create a compute context!\n");
        return EXIT_FAILURE;
    }

    // Create a command commands
    //
    commands = clCreateCommandQueue(context, device_id, 0, &err);
    if (!commands)
    {
        printf("Error: Failed to create a command commands!\n");
        return EXIT_FAILURE;
    }

    // Create the compute program from the source buffer
    //
    program = clCreateProgramWithSource(context, 1, (const char **) & KernelSource, NULL, &err);
    if (!program)
    {
        printf("Error: Failed to create compute program!\n");
        return EXIT_FAILURE;
    }

    // Build the program executable
    //
    err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
    if (err != CL_SUCCESS)
    {
        size_t len;
        char buffer[2048];

        printf("Error: Failed to build program executable!\n");
        clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
        printf("%s\n", buffer);
        exit(1);
    }

    // Create the compute kernel in the program we wish to run
    //
    kernel = clCreateKernel(program, "square", &err);
    if (!kernel || err != CL_SUCCESS)
    {
        printf("Error: Failed to create compute kernel!\n");
        exit(1);
    }

    // Create the input and output arrays in device memory for our calculation
    //
    input = clCreateBuffer(context,  CL_MEM_READ_ONLY,  sizeof(float) * count, NULL, NULL);
    output = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float) * count, NULL, NULL);
    if (!input || !output)
    {
        printf("Error: Failed to allocate device memory!\n");
        exit(1);
    }

    cl_uint query_id;
    cl_uint next_query_id;
    cl_perf_query_intel query;
    
    cl_get_first_perf_query_id = (clGetFirstPerfQueryIdIntel_fn)
      clGetExtensionFunctionAddressForPlatform(platform_id, "clGetFirstPerfQueryIdIntel");
    if (!cl_get_first_perf_query_id)
      printf("error: failed to get clGetFirstPerfQueryIdIntel func\n");

    cl_get_next_perf_query_id = (clGetNextPerfQueryIdIntel_fn)
      clGetExtensionFunctionAddressForPlatform(platform_id, "clGetNextPerfQueryIdIntel");
    if (!cl_get_next_perf_query_id)
      printf("error: failed to get clGetNextPerfQueryIdIntel func\n");

    cl_get_perf_query_info = (clGetPerfQueryInfoIntel_fn)
      clGetExtensionFunctionAddressForPlatform(platform_id, "clGetPerfQueryInfoIntel");
    if (!cl_get_perf_query_info)
      printf("error: failed to get clGetPerfQueryInfoIntel func\n");

    cl_get_perf_counter_info = (clGetPerfCounterInfoIntel_fn)
      clGetExtensionFunctionAddressForPlatform(platform_id, "clGetPerfCounterInfoIntel");
    if (!cl_get_perf_counter_info)
      printf("error: failed to get clGetPerfCounterInfoIntel func\n");

    cl_create_perf_query = (clCreatePerfQueryIntel_fn)
      clGetExtensionFunctionAddressForPlatform(platform_id,
					       "clCreatePerfQueryIntel");
    if (!cl_create_perf_query)
      printf("error: failed to get clCreatePerfQueryIntel func\n");

    cl_delete_perf_query = (clDeletePerfQueryIntel_fn)
      clGetExtensionFunctionAddressForPlatform(platform_id,
					       "clDeletePerfQueryIntel");
    if (!cl_delete_perf_query)
      printf("error: failed to get clDeletePerfQueryIntel func\n");

    cl_begin_perf_query = (clBeginPerfQueryIntel_fn)
      clGetExtensionFunctionAddressForPlatform(platform_id,
					       "clBeginPerfQueryIntel");
    if (!cl_begin_perf_query)
      printf("error: failed to get clBeginPerfQueryIntel func\n");

    cl_end_perf_query = (clEndPerfQueryIntel_fn)
      clGetExtensionFunctionAddressForPlatform(platform_id,
					       "clEndPerfQueryIntel");
    if (!cl_end_perf_query)
      printf("error: failed to get clEndPerfQueryIntel func\n");

    cl_get_perf_query_data = (clGetPerfQueryDataIntel_fn)
      clGetExtensionFunctionAddressForPlatform(platform_id,
					       "clGetPerfQueryDataIntel");
    if (!cl_get_perf_query_data)
      printf("error: failed to get clGetPerfQueryDataIntel func\n");

    cl_get_first_perf_query_id(context, &query_id);
    cl_get_next_perf_query_id(context, query_id, &next_query_id);
    printf("first query: %d, next query %d\n", query_id, next_query_id);

    cl_char query_name[100];
    cl_uint data_size, n_counter, n_instance;
    char *perf_data;
    cl_uint written;

    cl_get_perf_query_info(context, query_id, 100, query_name, &data_size, &n_counter, &n_instance);
    printf("query name: %s\n", query_name);
    printf("query data size: %d\n", data_size);
    printf("query n_counter: %d\n", n_counter);
    printf("query n_instance: %d\n", n_instance);

    perf_data = (char *)malloc(data_size);
    
    struct perf_counter *c = (struct perf_counter *)malloc(n_counter * sizeof(struct perf_counter));
    if (!c)
      printf("alloc perf counter array fail\n");

    for (i = 0; i < n_counter; i++) {
      cl_get_perf_counter_info(context, query_id, i,
			       60, c[i].name,
			       120, c[i].desc,
			       &c[i].offset,
			       &c[i].size,
			       &c[i].type,
			       &c[i].data_type,
			       &c[i].raw_max);
      printf("counter: %d\n", i);
      printf("\tname: %s\n", c[i].name);
      printf("\tdesc: %s\n", c[i].desc);
      printf("\toffset: %d\n", c[i].offset);
      printf("\tsize: %d\n", c[i].size);
      printf("\ttype: 0x%x\n", c[i].type);
      printf("\tdata_type: 0x%x\n", c[i].data_type);
      printf("\traw_max: %lu\n", c[i].raw_max);
    }

    cl_create_perf_query(context, query_id, &query);

    // Write our data set into the input array in device memory 
    //
    err = clEnqueueWriteBuffer(commands, input, CL_TRUE, 0, sizeof(float) * count, data, 0, NULL, NULL);
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to write to source array!\n");
        exit(1);
    }

    // Set the arguments to our compute kernel
    //
    err = 0;
    err  = clSetKernelArg(kernel, 0, sizeof(cl_mem), &input);
    err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &output);
    err |= clSetKernelArg(kernel, 2, sizeof(unsigned int), &count);
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to set kernel arguments! %d\n", err);
        exit(1);
    }

    // Get the maximum work group size for executing the kernel on the device
    //
    err = clGetKernelWorkGroupInfo(kernel, device_id, CL_KERNEL_WORK_GROUP_SIZE, sizeof(local), &local, NULL);
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to retrieve kernel work group info! %d\n", err);
        exit(1);
    }

    cl_begin_perf_query(context, query);
    
    // Execute the kernel over the entire range of our 1d input data set
    // using the maximum number of work group items for this device
    //
    global = count;
    err = clEnqueueNDRangeKernel(commands, kernel, 1, NULL, &global, &local, 0, NULL, NULL);
    if (err)
    {
        printf("Error: Failed to execute kernel!\n");
        return EXIT_FAILURE;
    }

    cl_end_perf_query(context, query);

    // Wait for the command commands to get serviced before reading back results
    //
    clFinish(commands);

    cl_get_perf_query_data(context, query,
			   0, data_size, perf_data, &written);

    printf("perf data written: %d\n", written);
#if 0
    for (i = 0; i < data_size; i++)
      printf("%x ", perf_data[i]);
    printf("\n");
#endif
    
    for (i = 0; i < n_counter; i++) {
      union {
	uint32_t a;
	uint64_t b;
	float f;
	double d;
      } val;
      printf("counter %s: ", c[i].name);
      switch (c[i].data_type) {
      case PERFQUERY_COUNTER_DATA_UINT32_INTEL:
	val.a = *(uint32_t *)(perf_data + c[i].offset);
	printf("%u\n", val.a);
	break;
      case PERFQUERY_COUNTER_DATA_UINT64_INTEL:
	val.b = *(uint64_t *)(perf_data + c[i].offset);
	printf("%lu\n", val.b);
	break;
      case PERFQUERY_COUNTER_DATA_FLOAT_INTEL:
	val.f = *(float *)(perf_data + c[i].offset);
	printf("%f\n", val.f);
	break;
      case PERFQUERY_COUNTER_DATA_DOUBLE_INTEL:
	val.d = *(double *)(perf_data + c[i].offset);
	printf("%g\n", val.d);
	break;
      default:
	printf("unhandled data type: 0x%x\n", c[i].data_type);
	break;
      }
    }
    free(perf_data);
    cl_delete_perf_query(context, query);
    
    // Read back the results from the device to verify the output
    //
    err = clEnqueueReadBuffer( commands, output, CL_TRUE, 0, sizeof(float) * count, results, 0, NULL, NULL );  
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to read output array! %d\n", err);
        exit(1);
    }
    
    // Validate our results
    //
    correct = 0;
    for(i = 0; i < count; i++)
    {
        if(results[i] == data[i] * data[i])
            correct++;
    }
    
    // Print a brief summary detailing the results
    //
    printf("Computed '%d/%d' correct values!\n", correct, count);
    
    // Shutdown and cleanup
    //
    clReleaseMemObject(input);
    clReleaseMemObject(output);
    clReleaseProgram(program);
    clReleaseKernel(kernel);
    clReleaseCommandQueue(commands);
    clReleaseContext(context);

    return 0;
}


