@group(0) @binding(0) var curLayer : texture_storage_2d<rgba8unorm, read>;
@group(0) @binding(1) var kernel : texture_storage_3d<rgba32float, read>; // depth is 4
@group(0) @binding(2) var nextLayer : texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(3) var offset : vec2<i32>;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) GlobalInvocationID : vec3<u32>) {
	let pixelCoord : vec2<u32> = GlobalInvocationID.xy;

	let kernelXRadius : u32 = textureDimensions(kernel).x / 2u;
	let kernelYRadius : u32 = textureDimensions(kernel).y / 2u;

	let kernelRadius : vec2<u32> = vec2<u32>(kernelXRadius, kernelYRadius);
	let offset : vec2<u32> = offset + pixelCoord - kernelRadius;

	var sum : vec4<f32> = vec4<f32>(0.0, 0.0, 0.0, 0.0);
	for (var i : u32 = 0u; i < textureDimensions(kernel).x; i = i + 1u) {
		for (var j : u32 = 0u; j < textureDimensions(kernel).y; j = j + 1u) {
			let coord : vec2<u32> = vec2<u32>(i, j);
			let imageCoord : vec2<u32> = coord + offset;
			var pixel : vec4<f32> = vec4<f32>(0.0, 0.0, 0.0, 0.0);
			if (imageCoord.x < textureDimensions(inImage).x && imageCoord.y < textureDimensions(inImage).y) {
				pixel = vec4<f32>(textureLoad(inImage, vec2<i32>(imageCoord)));
			}
			else { 
				let edgeCoord : vec2<u32> = vec2<u32>(clamp(imageCoord.x, 0u, textureDimensions(inImage).x - 1u), clamp(imageCoord.y, 0u, textureDimensions(inImage).y - 1u));
				pixel = vec4<f32>(textureLoad(inImage, vec2<i32>(edgeCoord)));
			}
			let kernelValue : mat4x4<f32> = mat4x4(
                vec4<f32>(textureLoad(kernel, vec3<i32>(coord, 0))),
                vec4<f32>(textureLoad(kernel, vec3<i32>(coord, 1))),
                vec4<f32>(textureLoad(kernel, vec3<i32>(coord, 2))),
                vec4<f32>(textureLoad(kernel, vec3<i32>(coord, 3)))
            );
			sum = sum + pixel * kernelValue;
		}
	}

 	// Sigmoid activation
    pixel = vec4<f32>(
        pixel.x / (1.0 + abs(pixel.x)),
        pixel.y / (1.0 + abs(pixel.y)),
        pixel.z / (1.0 + abs(pixel.z)),
        pixel.w / (1.0 + abs(pixel.w))
    );

	textureStore(outImage, vec2<i32>(pixelCoord), vec4<f32>(sum));
}