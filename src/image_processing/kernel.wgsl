@group(0) @binding(0) var<storage, read> inImage : texture_storage_2d<rgb8unorm, access::read>;
@group(0) @binding(1) var<uniform> kernel : array<array<vec3<f32>>>;
@group(0) @binding(2) var<uniform> kernelSize : u32;
@group(0) @binding(3) var<storage, write> outImage : texture_storage_2d<rgb8unorm, write>;

@compute @workgroup_size(1, 1, 1)
fn main(@builtin(global_invocation_id) GlobalInvocationID : vec3<u32>) {
	let pixelCoord : vec2<u32> = GlobalInvocationID.xy;
	let kernelRadius : u32 = kernelSize / 2u;
	var sum : vec3<f32> = vec3<f32>(0.0, 0.0, 0.0);
	for (var i : u32 = 0u; i < kernelSize; i = i + 1u) {
		for (var j : u32 = 0u; j < kernelSize; j = j + 1u) {
			let coord : vec2<u32> = vec2<u32>(i, j);
			let imageCoord : vec2<u32> = pixelCoord + coord - vec2<u32>(kernelRadius, kernelRadius);
			let pixel : vec3<f32> =
				if (imageCoord.x < textureDimensions(inImage).x && imageCoord.y < textureDimensions(inImage).y)
					{ textureLoad(inImage, vec2<i32>(imageCoord), 0).rgb}
				else { vec3<f32>(0.0, 0.0, 0.0) };
			sum = sum + kernel[i][j] * pixel;
		}
	}
	textureStore(outImage, vec2<i32>(pixelCoord), vec4<f32>(sum, 1.0));
}
