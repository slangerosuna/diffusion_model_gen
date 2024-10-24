@group(0) @binding(0)
var curLayer : texture_storage_2d<rgba16unorm, read>;
@group(0) @binding(1)
var kernel : texture_storage_3d<rgba32float, read>; // depth is 4
@group(0) @binding(2)
var nextLayer : texture_storage_2d<rgba16unorm, write>;
@group(0) @binding(3)
var<uniform> layerOffset : vec2<i32>;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) GlobalInvocationID : vec3<u32>) {
	let pixelCoord : vec2<u32> = GlobalInvocationID.xy;

	let kernelXRadius : u32 = textureDimensions(kernel).x / 2u;
	let kernelYRadius : u32 = textureDimensions(kernel).y / 2u;

	let kernelRadius : vec2<u32> = vec2<u32>(kernelXRadius, kernelYRadius);
	let offset : vec2<i32> = layerOffset + vec2<i32>(pixelCoord) - vec2<i32>(kernelRadius);

	var sum : vec4<f32> = vec4<f32>(0.0, 0.0, 0.0, 0.0);
	for (var i : u32 = 0u; i < textureDimensions(kernel).x; i = i + 1u) {
		for (var j : u32 = 0u; j < textureDimensions(kernel).y; j = j + 1u) {
			let coord : vec2<u32> = vec2<u32>(i, j);
			let signedImageCoord : vec2<i32> = vec2<i32>(coord) + offset;
			if (signedImageCoord.x < 0 || signedImageCoord.y < 0) { continue; }
			let imageCoord : vec2<u32> = vec2<u32>(signedImageCoord);

			let edgeCoord : vec2<u32> = vec2<u32>(
				clamp(imageCoord.x, 0u, textureDimensions(curLayer).x - 1u),
				clamp(imageCoord.y, 0u, textureDimensions(curLayer).y - 1u),
			);
			let pixel = (vec4<f32>(textureLoad(curLayer, vec2<i32>(edgeCoord))) - vec4<f32>(0.5, 0.5, 0.5, 0.5)) * vec4<f32>(256, 256, 256, 256);

			let kernelValue : mat4x4<f32> = mat4x4<f32>(
                vec4<f32>(textureLoad(kernel, vec3<u32>(coord, 0))),
                vec4<f32>(textureLoad(kernel, vec3<u32>(coord, 1))),
                vec4<f32>(textureLoad(kernel, vec3<u32>(coord, 2))),
                vec4<f32>(textureLoad(kernel, vec3<u32>(coord, 3))),
            );
			sum = sum + kernelValue * pixel;
		}
	}

	let leakyRelu : f32 = 1./16.;
    sum = vec4<f32>(
		max(0.0, sum.x) + leakyRelu * min(0.0, sum.x),
		max(0.0, sum.y) + leakyRelu * min(0.0, sum.y),
		max(0.0, sum.z) + leakyRelu * min(0.0, sum.z),
		max(0.0, sum.w) + leakyRelu * min(0.0, sum.w),
    );
	sum = sum * vec4<f32>(1 / 256, 1 / 256, 1 / 256, 1 / 256) + vec4<f32>(0.5, 0.5, 0.5, 0.5);

	textureStore(nextLayer, vec2<i32>(pixelCoord), vec4<f32>(sum));
}
