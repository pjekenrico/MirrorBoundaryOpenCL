// Mirrored repeat works like this (1D example):
//   For an image of size 4 (indices: 0,1,2,3):
//   Input indices:  -3 -2 -1  0  1  2  3  4  5  6  7 ...
//   Mirrored map:    2  1  0  0  1  2  3  3  2  1  0 ...
//
// Essentially, the image is repeated infinitely in both directions,
// but every second repetition is mirrored (flipped).
//
static inline int2 mirrored_repeat_coords(int2 in_coords, int width, int height) {
	// --- X direction ---
	// Step 1: Handle negative coordinates by reflecting them:
	//   e.g., -1 → 0, -2 -> 1, -3 -> 2, etc.
	int x_prime = in_coords.x;
	if (x_prime < 0) x_prime = -x_prime - 1;

	// Step 2: Fold the coordinate into the [0, 2*width) range
	// This creates a repeating "sawtooth" pattern over twice the image width.
	int x_mod_2s = x_prime % (width * 2);

	// Step 3: If the folded value is inside [0, width), use it directly.
	// Otherwise, mirror it back into [0, width).
	int x_src = (x_mod_2s < width) ? x_mod_2s : (width * 2 - 1 - x_mod_2s);

	// --- Y direction (same logic as X) ---
	int y_prime = in_coords.y;
	if (y_prime < 0) y_prime = -y_prime - 1;
	int y_mod_2s = y_prime % (height * 2);
	int y_src = (y_mod_2s < height) ? y_mod_2s : (height * 2 - 1 - y_mod_2s);

	// Return the valid mirrored coordinates inside the source image
	return (int2)(x_src, y_src);
}

// This kernel uses the custom mirrored_repeat_coords function to get the
// correct coordinates for sampling, bypassing buggy driver implementations.
kernel void test_manual_mirrored_repeat(read_only image2d_t src_image, write_only image2d_t dst_image, const uint2 offset) {
	// We use a basic sampler with clamp-to-edge as a fallback, as our manual
	// function guarantees the coordinates are within the source image bounds.
	const sampler_t image_sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;

	// Get the coordinates of the current work-item for the output image
	int2 coords_out = (int2)(get_global_id(0), get_global_id(1));

	// Manually calculate the mirrored source coordinates
	int2 coords_in = mirrored_repeat_coords(coords_out - convert_int2(offset), get_image_width(src_image), get_image_height(src_image));

	// Read pixel value from the source image using the manual coordinates
	float4 pixel = read_imagef(src_image, image_sampler, coords_in);

	// Write the sampled pixel to the output image
	write_imagef(dst_image, coords_out, pixel);
}