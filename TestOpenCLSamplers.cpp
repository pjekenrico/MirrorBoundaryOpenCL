#include <vector>
#include <string>
#include <iomanip>
#include <sstream>
#include <fstream>
#include <iostream>

#define CL_TARGET_OPENCL_VERSION 120
#include <CL/cl.h>

// Macro to check OpenCL errors and print a message
#define CL_CHECK(status) \
    if (status != CL_SUCCESS) { \
        std::cerr << "OpenCL Error: " << status << " at line " << __LINE__ << std::endl; \
        exit(1); \
    }

// A simple CPU reference function to verify OpenCL results
// This function manually implements the CL_ADDRESS_MIRRORED_REPEAT behavior.
static void cpu_reference_calculation(
	const std::vector<float>& src_data,
	std::vector<float>& dst_data,
	int src_w, int src_h,
	int dst_w, int dst_h,
	unsigned int offset_x, unsigned int offset_y // <-- new parameters
) {
	for (int y_out = 0; y_out < dst_h; ++y_out) {
		for (int x_out = 0; x_out < dst_w; ++x_out) {

			// Apply offset (same as in kernel: coords_shifted = coords_out - offset)
			int x_shifted = x_out - static_cast<int>(offset_x);
			int y_shifted = y_out - static_cast<int>(offset_y);

			// Mirroring logic for X coordinate
			int x_prime = x_shifted;
			if (x_prime < 0) x_prime = -x_prime - 1;
			int x_mod_2s = x_prime % (src_w * 2);
			int x_src = (x_mod_2s < src_w) ? x_mod_2s : (src_w * 2 - 1 - x_mod_2s);

			// Mirroring logic for Y coordinate
			int y_prime = y_shifted;
			if (y_prime < 0) y_prime = -y_prime - 1;
			int y_mod_2s = y_prime % (src_h * 2);
			int y_src = (y_mod_2s < src_h) ? y_mod_2s : (src_h * 2 - 1 - y_mod_2s);

			// Read the value from the source data
			dst_data[y_out * dst_w + x_out] = src_data[y_src * src_w + x_src];
		}
	}
}

template<typename T>
static void printMatrix(const std::vector<T>& mat, size_t width, size_t height) {
	// Determine the maximum width of any element
	size_t maxWidth = 0;
	for (auto& val : mat) {
		std::ostringstream oss;
		oss << val;
		maxWidth = std::max(maxWidth, oss.str().size());
	}

	std::cout << "[\n";
	for (size_t y = 0; y < height; ++y) {
		std::cout << " [";
		for (size_t x = 0; x < width; ++x) {
			std::cout << std::setw(maxWidth) << mat[y * width + x];
			if (x + 1 < width) std::cout << " ";
		}
		std::cout << "]";
		if (y + 1 < height) std::cout << ",";
		std::cout << "\n";
	}
	std::cout << "]\n";
}

int main() {
	// --- 1. Define data and image dimensions ---
	const int input_width = 4;
	const int input_height = 5;
	const int output_width = 12;
	const int output_height = 12;
	const int input_size = input_width * input_height;
	const int output_size = output_width * output_height;
	cl_uint offset[2] = { 5, 2 }; // Offset to shift the sampling coordinates and see the mirror boundary effects on all image boundaries

	// Create a simple host data array for the input image
	std::vector<float> host_input_data(input_size);
	for (int i = 0; i < input_size; ++i) {
		host_input_data[i] = (float)i;
	}

	// Allocate memory for the results from the CPU and the OpenCL kernel
	std::vector<float> cpu_result(output_size);
	std::vector<float> kernel_result_manual(output_size);

	// --- 2. Perform CPU reference calculation ---
	cpu_reference_calculation(host_input_data, cpu_result, input_width, input_height, output_width, output_height, offset[0], offset[1]);

	// --- 3. OpenCL setup ---
	cl_platform_id platform;
	cl_device_id device;
	cl_context context;
	cl_command_queue queue;
	cl_program program;
	cl_kernel kernel_manual;
	cl_mem image_in, image_out_manual;
	cl_int err;

	// Get platform and device
	CL_CHECK(clGetPlatformIDs(1, &platform, NULL));
	CL_CHECK(clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL));

	// Create context and command queue
	context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
	CL_CHECK(err);
	queue = clCreateCommandQueue(context, device, 0, &err);
	CL_CHECK(err);

	// --- 4. Load and build the OpenCL kernel source ---
	std::ifstream kernel_file("sampler_test.cl");
	if (!kernel_file.is_open()) {
		std::cerr << "Could not open kernel file sampler_test.cl" << std::endl;
		return 1;
	}
	std::string kernel_source_str((std::istreambuf_iterator<char>(kernel_file)),
		std::istreambuf_iterator<char>());
	const char* kernel_source = kernel_source_str.c_str();

	program = clCreateProgramWithSource(context, 1, &kernel_source, NULL, &err);
	CL_CHECK(err);

	// Build the program executable
	err = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
	if (err != CL_SUCCESS) {
		size_t log_size;
		CL_CHECK(clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size));
		std::vector<char> log_buffer(log_size);
		CL_CHECK(clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, log_buffer.data(), NULL));
		std::cerr << "OpenCL Build Error: " << log_buffer.data() << std::endl;
		return 1;
	}

	// Create the new kernel
	kernel_manual = clCreateKernel(program, "test_manual_mirrored_repeat", &err);
	CL_CHECK(err);

	// --- 5. Create memory objects on the device ---
	// Image format and description
	cl_image_format image_format{};
	image_format.image_channel_order = CL_R;
	image_format.image_channel_data_type = CL_FLOAT;

	// Input image descriptor
	cl_image_desc input_image_desc{};
	input_image_desc.image_type = CL_MEM_OBJECT_IMAGE2D;
	input_image_desc.image_width = input_width;
	input_image_desc.image_height = input_height;
	input_image_desc.image_row_pitch = input_width * sizeof(float);

	// Create the input image object
	image_in = clCreateImage(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
		&image_format, &input_image_desc, host_input_data.data(), &err);
	CL_CHECK(err);

	// Output image descriptor
	cl_image_desc output_image_desc{};
	output_image_desc.image_type = CL_MEM_OBJECT_IMAGE2D;
	output_image_desc.image_width = output_width;
	output_image_desc.image_height = output_height;

	// Create the output image object
	image_out_manual = clCreateImage(context, CL_MEM_WRITE_ONLY, &image_format, &output_image_desc, NULL, &err);
	CL_CHECK(err);

	// --- 6. Set kernel arguments and enqueue kernel ---
	CL_CHECK(clSetKernelArg(kernel_manual, 0, sizeof(cl_mem), &image_in));
	CL_CHECK(clSetKernelArg(kernel_manual, 1, sizeof(cl_mem), &image_out_manual));
	CL_CHECK(clSetKernelArg(kernel_manual, 2, sizeof(offset), offset));

	// The global work size is based on the larger output image
	size_t global_work_size[2] = { (size_t)output_width, (size_t)output_height };
	CL_CHECK(clEnqueueNDRangeKernel(queue, kernel_manual, 2, NULL, global_work_size, NULL, 0, NULL, NULL));

	// --- 7. Read results back to host memory ---
	size_t origin[3] = { 0, 0, 0 };
	size_t region[3] = { (size_t)output_width, (size_t)output_height, 1 };

	CL_CHECK(clEnqueueReadImage(queue, image_out_manual, CL_TRUE, origin, region,
		output_width * sizeof(float), 0, kernel_result_manual.data(), 0, NULL, NULL));

	// Wait for all commands to finish
	CL_CHECK(clFinish(queue));

	// --- 8. Compare results ---
	bool manual_match = true;
	for (int i = 0; i < output_size; ++i) {
		if (cpu_result[i] != kernel_result_manual[i]) {
			manual_match = false;
			std::cerr << "Mismatch found at index " << i << ": CPU=" << cpu_result[i] << ", Kernel=" << kernel_result_manual[i] << std::endl;
			break;
		}
	}

	std::cout << "\n--- Results Comparison ---" << std::endl;
	std::cout << "CPU vs. Manual Mirrored Kernel: " << (manual_match ? "PASS" : "FAIL") << std::endl;
	//if (!manual_match)
	{
		std::cout << "CPU Result:" << std::endl;
		printMatrix(cpu_result, output_width, output_height);
		std::cout << "\nKernel Result:" << std::endl;
		printMatrix(kernel_result_manual, output_width, output_height);
	}

	// --- 9. Cleanup ---
	clReleaseMemObject(image_in);
	clReleaseMemObject(image_out_manual);
	clReleaseKernel(kernel_manual);
	clReleaseProgram(program);
	clReleaseCommandQueue(queue);
	clReleaseContext(context);

	return 0;
}
