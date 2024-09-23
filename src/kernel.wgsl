@group(0) @binding(0) var inImage : texture_storage_2d<rgba8unorm, read>;
@group(0) @binding(1) var kernel : texture_storage_2d<rgba16float, read>;
@group(0) @binding(2) var outImage : texture_storage_2d<rgba8unorm, write>;

@compute @workgroup_size(1, 1, 1)
fn main(@builtin(global_invocation_id) GlobalInvocationID : vec3<u32>) {
	let pixelCoord : vec2<u32> = GlobalInvocationID.xy;

	let kernelXRadius : u32 = textureDimensions(kernel).x / 2u;
	let kernelYRadius : u32 = textureDimensions(kernel).y / 2u;

	let kernelRadius : vec2<u32> = vec2<u32>(kernelXRadius, kernelYRadius);
	let offset : vec2<u32> = pixelCoord - kernelRadius;

	var sum : vec4<f32> = vec4<f32>(0.0, 0.0, 0.0, 0.0);
	for (var i : u32 = 0u; i < textureDimensions(kernel).x; i = i + 1u) {
		for (var j : u32 = 0u; j < textureDimensions(kernel).y; j = j + 1u) {
			let coord : vec2<u32> = vec2<u32>(i, j);
			let imageCoord : vec2<u32> = coord + offset;
			var pixel : vec4<f32> = vec4<f32>(0.0, 0.0, 0.0, 0.0);
			if (imageCoord.x < textureDimensions(inImage).x && imageCoord.y < textureDimensions(inImage).y) {
				pixel = vec4<f32>(textureLoad(inImage, vec2<i32>(imageCoord)));
			}
			else { pixel = vec4<f32>(0.0, 0.0, 0.0, 0.0); }
			let kernelValue : vec4<f32> = vec4<f32>(textureLoad(kernel, coord));
			sum = sum + pixel * kernelValue;
		}
	}

	textureStore(outImage, vec2<i32>(pixelCoord), vec4<f32>(sum));
}