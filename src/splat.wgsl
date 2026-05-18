struct Uniforms {
    view: mat4x4<f32>,
    projection: mat4x4<f32>,
    camera_position: vec4<f32>,
    viewport_size: vec2<f32>,
    near_plane: f32,
    _padding: f32,
};

struct SphericalHarmonics {
    values: array<f32>,
};

struct Gaussian {
    centroid: vec3<f32>,
    opacity: f32,
    scale: vec3<f32>,
    _padding: f32,
    rotation: vec4<f32>,
};

struct Gaussians {
    values: array<Gaussian>,
};

struct SortedIndices {
    values: array<u32>,
};

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read> gaussians: Gaussians;
@group(0) @binding(2) var<storage, read> spherical_harmonics: SphericalHarmonics;
@group(0) @binding(3) var<storage, read> sorted_indices: SortedIndices;

struct VertexInput {
    @builtin(vertex_index) vertex_index: u32,
    @builtin(instance_index) instance_index: u32,
};

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) splat_coord: vec2<f32>,
    @location(1) splat_color: vec3<f32>,
    @location(2) splat_opacity: f32,
};

const SH_FLOAT_COUNT: u32 = 48u;
const SPLAT_EXTENT: f32 = 3.0;
const MAX_SPLAT_RADIUS_PX: f32 = 64.0;

fn quad_corner(vertex_index: u32) -> vec2<f32> {
    if (vertex_index == 0u) {
        return vec2<f32>(-1.0, -1.0);
    }
    if (vertex_index == 1u) {
        return vec2<f32>(1.0, -1.0);
    }
    if (vertex_index == 2u) {
        return vec2<f32>(-1.0, 1.0);
    }
    return vec2<f32>(1.0, 1.0);
}

const SH_C0: f32 = 0.28209479177387814;
const SH_C1: f32 = 0.4886025119029199;
const SH_C2_0: f32 = 1.0925484305920792;
const SH_C2_1: f32 = -1.0925484305920792;
const SH_C2_2: f32 = 0.31539156525252005;
const SH_C2_3: f32 = -1.0925484305920792;
const SH_C2_4: f32 = 0.5462742152960396;
const SH_C3_0: f32 = -0.5900435899266435;
const SH_C3_1: f32 = 2.890611442640554;
const SH_C3_2: f32 = -0.4570457994644658;
const SH_C3_3: f32 = 0.3731763325901154;
const SH_C3_4: f32 = -0.4570457994644658;
const SH_C3_5: f32 = 1.445305721320277;
const SH_C3_6: f32 = -0.5900435899266435;

fn spherical_harmonic_float(instance_index: u32, index: u32) -> f32 {
    let splat_index = sorted_indices.values[instance_index];
    return spherical_harmonics.values[splat_index * SH_FLOAT_COUNT + index];
}

fn spherical_harmonic_coefficient(instance_index: u32, coefficient: u32) -> vec3<f32> {
    let base_index = coefficient * 3u;
    return vec3<f32>(
        spherical_harmonic_float(instance_index, base_index),
        spherical_harmonic_float(instance_index, base_index + 1u),
        spherical_harmonic_float(instance_index, base_index + 2u),
    );
}

fn spherical_harmonics_to_rgb(instance_index: u32, direction: vec3<f32>) -> vec3<f32> {
    let x = direction.x;
    let y = direction.y;
    let z = direction.z;

    var rgb = SH_C0 * spherical_harmonic_coefficient(instance_index, 0u);

    rgb += -SH_C1 * y * spherical_harmonic_coefficient(instance_index, 1u);
    rgb += SH_C1 * z * spherical_harmonic_coefficient(instance_index, 2u);
    rgb += -SH_C1 * x * spherical_harmonic_coefficient(instance_index, 3u);

    rgb += SH_C2_0 * x * y * spherical_harmonic_coefficient(instance_index, 4u);
    rgb += SH_C2_1 * y * z * spherical_harmonic_coefficient(instance_index, 5u);
    rgb += SH_C2_2 * (2.0 * z * z - x * x - y * y) * spherical_harmonic_coefficient(instance_index, 6u);
    rgb += SH_C2_3 * x * z * spherical_harmonic_coefficient(instance_index, 7u);
    rgb += SH_C2_4 * (x * x - y * y) * spherical_harmonic_coefficient(instance_index, 8u);

    rgb += SH_C3_0 * y * (3.0 * x * x - y * y) * spherical_harmonic_coefficient(instance_index, 9u);
    rgb += SH_C3_1 * x * y * z * spherical_harmonic_coefficient(instance_index, 10u);
    rgb += SH_C3_2 * y * (4.0 * z * z - x * x - y * y) * spherical_harmonic_coefficient(instance_index, 11u);
    rgb += SH_C3_3 * z * (2.0 * z * z - 3.0 * x * x - 3.0 * y * y) *
        spherical_harmonic_coefficient(instance_index, 12u);
    rgb += SH_C3_4 * x * (4.0 * z * z - x * x - y * y) * spherical_harmonic_coefficient(instance_index, 13u);
    rgb += SH_C3_5 * z * (x * x - y * y) * spherical_harmonic_coefficient(instance_index, 14u);
    rgb += SH_C3_6 * x * (x * x - 3.0 * y * y) * spherical_harmonic_coefficient(instance_index, 15u);

    return clamp(rgb + vec3<f32>(0.5), vec3<f32>(0.0), vec3<f32>(1.0));
}

fn quaternion_to_mat3(quaternion: vec4<f32>) -> mat3x3<f32> {
    let q = normalize(quaternion);
    let w = q.x;
    let x = q.y;
    let y = q.z;
    let z = q.w;

    return mat3x3<f32>(
        vec3<f32>(1.0 - 2.0 * y * y - 2.0 * z * z, 2.0 * x * y + 2.0 * w * z, 2.0 * x * z - 2.0 * w * y),
        vec3<f32>(2.0 * x * y - 2.0 * w * z, 1.0 - 2.0 * x * x - 2.0 * z * z, 2.0 * y * z + 2.0 * w * x),
        vec3<f32>(2.0 * x * z + 2.0 * w * y, 2.0 * y * z - 2.0 * w * x, 1.0 - 2.0 * x * x - 2.0 * y * y),
    );
}

fn build_covariance_3d(scale: vec3<f32>, rotation: vec4<f32>) -> mat3x3<f32> {
    let sigma = exp(scale);
    let rotation_matrix = quaternion_to_mat3(rotation);
    let scale_matrix = mat3x3<f32>(
        vec3<f32>(sigma.x, 0.0, 0.0),
        vec3<f32>(0.0, sigma.y, 0.0),
        vec3<f32>(0.0, 0.0, sigma.z),
    );
    let rs = rotation_matrix * scale_matrix;
    let covariance = rs * transpose(rs);

    let preview_flip_y = mat3x3<f32>(
        vec3<f32>(1.0, 0.0, 0.0),
        vec3<f32>(0.0, -1.0, 0.0),
        vec3<f32>(0.0, 0.0, 1.0),
    );
    return preview_flip_y * covariance * preview_flip_y;
}

fn project_covariance_to_screen(center_camera: vec3<f32>, covariance_3d: mat3x3<f32>) -> mat3x3<f32> {
    let focal_x = uniforms.projection[0][0] * uniforms.viewport_size.x * 0.5;
    let focal_y = uniforms.projection[1][1] * uniforms.viewport_size.y * 0.5;
    let z = center_camera.z;

    let jacobian = mat3x3<f32>(
        vec3<f32>(-focal_x / z, 0.0, 0.0),
        vec3<f32>(0.0, -focal_y / z, 0.0),
        vec3<f32>(focal_x * center_camera.x / (z * z), focal_y * center_camera.y / (z * z), 0.0),
    );

    let view_linear = mat3x3<f32>(
        uniforms.view[0].xyz,
        uniforms.view[1].xyz,
        uniforms.view[2].xyz,
    );
    let transform = jacobian * view_linear;
    return transform * covariance_3d * transpose(transform);
}

@vertex
fn vs_main(input: VertexInput) -> VertexOutput {
    let splat_index = sorted_indices.values[input.instance_index];
    let gaussian = gaussians.values[splat_index];
    let corner = quad_corner(input.vertex_index);
    let center_clip = uniforms.projection * uniforms.view * vec4<f32>(gaussian.centroid, 1.0);
    let center_camera = (uniforms.view * vec4<f32>(gaussian.centroid, 1.0)).xyz;

    var output: VertexOutput;
    if (center_clip.w <= 0.0 || center_camera.z >= -uniforms.near_plane ||
        uniforms.viewport_size.x <= 0.0 || uniforms.viewport_size.y <= 0.0) {
        output.position = vec4<f32>(2.0, 2.0, 2.0, 1.0);
        output.splat_coord = corner;
        output.splat_color = vec3<f32>(0.0);
        output.splat_opacity = 0.0;
        return output;
    }

    let covariance_2d = project_covariance_to_screen(
        center_camera,
        build_covariance_3d(gaussian.scale, gaussian.rotation),
    );

    let a = covariance_2d[0][0] + 0.3;
    let b = covariance_2d[1][0];
    let d = covariance_2d[1][1] + 0.3;

    let determinant = a * d - b * b;
    let mid = 0.5 * (a + d);
    let radius = sqrt(max(mid * mid - determinant, 0.0));
    let lambda_1 = max(mid + radius, 0.01);
    let lambda_2 = max(mid - radius, 0.01);

    var axis_direction_1: vec2<f32>;
    if (abs(b) > 0.00001) {
        axis_direction_1 = normalize(vec2<f32>(b, lambda_1 - a));
    } else if (a >= d) {
        axis_direction_1 = vec2<f32>(1.0, 0.0);
    } else {
        axis_direction_1 = vec2<f32>(0.0, 1.0);
    }
    let axis_direction_2 = vec2<f32>(-axis_direction_1.y, axis_direction_1.x);

    let axis_1 = min(SPLAT_EXTENT * sqrt(lambda_1), MAX_SPLAT_RADIUS_PX) * axis_direction_1;
    let axis_2 = min(SPLAT_EXTENT * sqrt(lambda_2), MAX_SPLAT_RADIUS_PX) * axis_direction_2;

    let ndc = center_clip.xyz / center_clip.w;
    let pixel_offset = corner.x * axis_1 + corner.y * axis_2;
    let ndc_offset = pixel_offset / (uniforms.viewport_size * 0.5);

    output.position = vec4<f32>(ndc.xy + ndc_offset, ndc.z, 1.0);
    output.splat_coord = corner;
    output.splat_color = spherical_harmonics_to_rgb(
        input.instance_index,
        normalize(gaussian.centroid - uniforms.camera_position.xyz),
    );
    output.splat_opacity = 1.0 / (1.0 + exp(-gaussian.opacity));
    return output;
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    let power = -0.5 * SPLAT_EXTENT * SPLAT_EXTENT * dot(input.splat_coord, input.splat_coord);
    if (power < -9.0) {
        discard;
    }

    let alpha = min(0.99, input.splat_opacity * exp(power));
    if (alpha < 0.001) {
        discard;
    }

    return vec4<f32>(input.splat_color, alpha);
}
