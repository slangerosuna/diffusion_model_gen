@group(0) @binding(0)
var layer : texture_storage_2d<rgba16unorm, read_write>;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) GlobalInvocationID : vec3<u32>) {
    let pixelCoord : vec2<u32> = GlobalInvocationID.xy;

    let pixel : vec4<f32> = (vec4<f32>(textureLoad(layer, vec2<i32>(pixelCoord))) - vec4<f32>(0.5, 0.5, 0.5, 0.5)) * vec4<f32>(256, 256, 256, 256);
    let sigmoid : vec4<f32> = vec4<f32>(
        1.0 / (1.0 + exp(-pixel.x)),
        1.0 / (1.0 + exp(-pixel.y)),
        1.0 / (1.0 + exp(-pixel.z)),
        1.0 / (1.0 + exp(-pixel.w)),
    );
    let betweenZeroAndTwo : vec4<f32> = sigmoid + vec4<f32>(1.0, 1.0, 1.0, 1.0);
    let betweenZeroAndOne : vec4<f32> = betweenZeroAndTwo * vec4<f32>(0.5, 0.5, 0.5, 0.5);

    // output is quantized differently than input because its range is [0, 1] instead of [-127, 127]
    textureStore(layer, vec2<i32>(pixelCoord), vec4<f32>(betweenZeroAndOne));
}