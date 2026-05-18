const std = @import("std");
const zonk = @import("zonk");
const zmath = @import("zmath");
const splat = @import("splat.zig");

const zgpu = zonk.zgpu;
const wgpu = zgpu.wgpu;

const screen_width = 420;
const screen_height = 280;
const window_width = screen_width * 2;
const window_height = screen_height * 2;
const quad_vertex_count = 4;
const sort_interval_seconds = 1.0 / 12.0;

const Vec3 = zmath.F32x4;

const SceneBounds = struct {
    center: Vec3 = .{ 0.0, 0.0, 0.0, 0.0 },
    radius: f32 = 1.0,
};

const Camera = struct {
    position: Vec3,
    target: Vec3,
    up: Vec3 = .{ 0.0, 1.0, 0.0, 0.0 },
    fov_y_radians: f32 = std.math.degreesToRadians(45.0),
    near: f32,
    far: f32,

    fn view(self: Camera) zmath.Mat {
        return zmath.lookAtRh(self.position, self.target, self.up);
    }

    fn projection(self: Camera, aspect: f32) zmath.Mat {
        return zmath.perspectiveFovRh(self.fov_y_radians, aspect, self.near, self.far);
    }
};

const GpuGaussian = extern struct {
    centroid: [3]f32,
    opacity: f32,
    scale: [3]f32,
    padding: f32 = 0.0,
    rotation: [4]f32,
};

const SplatUniforms = extern struct {
    view: [16]f32,
    projection: [16]f32,
    camera_position: [4]f32,
    viewport_size: [2]f32,
    near_plane: f32,
    padding: f32 = 0.0,
};

const SplatSortEntry = struct {
    camera_z: f32,
    splat_index: usize,
};

const SplatDrawData = struct {
    gaussians: std.ArrayList(GpuGaussian) = .empty,
    spherical_harmonics: std.ArrayList(f32) = .empty,

    fn deinit(self: *SplatDrawData, gpa: std.mem.Allocator) void {
        self.gaussians.deinit(gpa);
        self.spherical_harmonics.deinit(gpa);
    }
};

const SplatRenderer = struct {
    allocator: std.mem.Allocator,
    pipeline: zgpu.RenderPipelineHandle,
    bind_group_layout: zgpu.BindGroupLayoutHandle,
    gaussian_buffer: zgpu.BufferHandle,
    spherical_harmonics_buffer: zgpu.BufferHandle,
    index_buffer: zgpu.BufferHandle,
    bind_group: zgpu.BindGroupHandle,
    instance_count: u32,

    fn init(allocator: std.mem.Allocator, gctx: *zgpu.GraphicsContext, draw_data: SplatDrawData) !SplatRenderer {
        const shader_module = zgpu.createWgslShaderModule(gctx.device, @embedFile("splat.wgsl"), "splat");
        defer shader_module.release();

        const bind_group_layout = gctx.createBindGroupLayout(&.{
            zgpu.bufferEntry(0, wgpu.ShaderStages.vertex, .uniform, true, @sizeOf(SplatUniforms)),
            zgpu.bufferEntry(1, wgpu.ShaderStages.vertex, .read_only_storage, false, @sizeOf(GpuGaussian)),
            zgpu.bufferEntry(2, wgpu.ShaderStages.vertex, .read_only_storage, false, splat.SH_FLOAT_COUNT * @sizeOf(f32)),
            zgpu.bufferEntry(3, wgpu.ShaderStages.vertex, .read_only_storage, false, @sizeOf(u32)),
        });
        errdefer gctx.releaseResource(bind_group_layout);

        const pipeline_layout = gctx.createPipelineLayout(&.{bind_group_layout});
        defer gctx.releaseResource(pipeline_layout);

        const color_blend = wgpu.BlendState{
            .color = .{
                .src_factor = .src_alpha,
                .dst_factor = .one_minus_src_alpha,
                .operation = .add,
            },
            .alpha = .{
                .src_factor = .one,
                .dst_factor = .one_minus_src_alpha,
                .operation = .add,
            },
        };
        const color_targets = [_]wgpu.ColorTargetState{.{
            .format = zgpu.GraphicsContext.swapchain_format,
            .blend = &color_blend,
        }};

        const pipeline = gctx.createRenderPipeline(pipeline_layout, .{
            .vertex = .{
                .module = shader_module,
                .entry_point = wgpu.compat.shaderEntryPoint("vs_main"),
                .buffer_count = 0,
                .buffers = null,
            },
            .primitive = .{
                .topology = .triangle_strip,
                .front_face = .ccw,
                .cull_mode = .none,
            },
            .fragment = &wgpu.FragmentState{
                .module = shader_module,
                .entry_point = wgpu.compat.shaderEntryPoint("fs_main"),
                .target_count = color_targets.len,
                .targets = &color_targets,
            },
            .depth_stencil = null,
            .multisample = .{},
        });
        errdefer gctx.releaseResource(pipeline);

        const gaussian_buffer = gctx.createBuffer(.{
            .label = wgpu.compat.bufferLabel("splat gaussians"),
            .usage = wgpu.combineBufferUsage(wgpu.BufferUsages.copy_dst, wgpu.BufferUsages.storage),
            .size = draw_data.gaussians.items.len * @sizeOf(GpuGaussian),
        });
        errdefer gctx.releaseResource(gaussian_buffer);

        const spherical_harmonics_buffer = gctx.createBuffer(.{
            .label = wgpu.compat.bufferLabel("splat sh"),
            .usage = wgpu.combineBufferUsage(wgpu.BufferUsages.copy_dst, wgpu.BufferUsages.storage),
            .size = draw_data.spherical_harmonics.items.len * @sizeOf(f32),
        });
        errdefer gctx.releaseResource(spherical_harmonics_buffer);

        const index_buffer = gctx.createBuffer(.{
            .label = wgpu.compat.bufferLabel("splat sorted indices"),
            .usage = wgpu.combineBufferUsage(wgpu.BufferUsages.copy_dst, wgpu.BufferUsages.storage),
            .size = draw_data.gaussians.items.len * @sizeOf(u32),
        });
        errdefer gctx.releaseResource(index_buffer);

        gctx.queue.writeBuffer(gctx.lookupResource(gaussian_buffer).?, 0, GpuGaussian, draw_data.gaussians.items);
        gctx.queue.writeBuffer(gctx.lookupResource(spherical_harmonics_buffer).?, 0, f32, draw_data.spherical_harmonics.items);

        const bind_group = gctx.createBindGroup(bind_group_layout, &.{
            .{
                .binding = 0,
                .buffer_handle = gctx.uniforms.buffer,
                .offset = 0,
                .size = @sizeOf(SplatUniforms),
            },
            .{
                .binding = 1,
                .buffer_handle = gaussian_buffer,
                .offset = 0,
                .size = draw_data.gaussians.items.len * @sizeOf(GpuGaussian),
            },
            .{
                .binding = 2,
                .buffer_handle = spherical_harmonics_buffer,
                .offset = 0,
                .size = draw_data.spherical_harmonics.items.len * @sizeOf(f32),
            },
            .{
                .binding = 3,
                .buffer_handle = index_buffer,
                .offset = 0,
                .size = draw_data.gaussians.items.len * @sizeOf(u32),
            },
        });
        errdefer gctx.releaseResource(bind_group);

        return .{
            .allocator = allocator,
            .pipeline = pipeline,
            .bind_group_layout = bind_group_layout,
            .gaussian_buffer = gaussian_buffer,
            .spherical_harmonics_buffer = spherical_harmonics_buffer,
            .index_buffer = index_buffer,
            .bind_group = bind_group,
            .instance_count = @intCast(draw_data.gaussians.items.len),
        };
    }

    fn uploadSortedIndices(
        self: *SplatRenderer,
        allocator: std.mem.Allocator,
        gctx: *zgpu.GraphicsContext,
        scene: splat.Scene,
        view: zmath.Mat,
    ) !void {
        var entries: std.ArrayList(SplatSortEntry) = .empty;
        defer entries.deinit(allocator);
        var indices: std.ArrayList(u32) = .empty;
        defer indices.deinit(allocator);

        try entries.ensureTotalCapacity(allocator, scene.splats.items.len);
        try indices.ensureTotalCapacity(allocator, scene.splats.items.len);

        for (scene.splats.items, 0..) |gaussian, index| {
            const camera_centroid = zmath.mul(zmath.loadArr3w(gaussian.centroid, 1.0), view);
            entries.appendAssumeCapacity(.{
                .camera_z = camera_centroid[2],
                .splat_index = index,
            });
        }

        std.sort.pdq(SplatSortEntry, entries.items, {}, lessThanSortEntry);
        for (entries.items) |entry| {
            indices.appendAssumeCapacity(@intCast(entry.splat_index));
        }

        gctx.queue.writeBuffer(gctx.lookupResource(self.index_buffer).?, 0, u32, indices.items);
    }

    fn deinit(self: *SplatRenderer, gctx: *zgpu.GraphicsContext) void {
        gctx.releaseResource(self.bind_group);
        gctx.releaseResource(self.gaussian_buffer);
        gctx.releaseResource(self.spherical_harmonics_buffer);
        gctx.releaseResource(self.index_buffer);
        gctx.releaseResource(self.pipeline);
        gctx.releaseResource(self.bind_group_layout);
    }

    fn draw(self: *SplatRenderer, gctx: *zgpu.GraphicsContext, uniforms: SplatUniforms) !void {
        const view = gctx.getCurrentTextureView();
        defer view.release();

        const encoder = gctx.device.createCommandEncoder(null);
        defer encoder.release();

        const color_attachments = [_]wgpu.RenderPassColorAttachment{.{
            .view = view,
            .load_op = .clear,
            .store_op = .store,
            .clear_value = .{ .r = 0.0, .g = 0.0, .b = 0.0, .a = 1.0 },
        }};
        const pass = encoder.beginRenderPass(.{
            .color_attachment_count = color_attachments.len,
            .color_attachments = &color_attachments,
        });

        const mem = gctx.uniformsAllocate(SplatUniforms, 1);
        if (mem.slice.len == 0) return error.UniformBufferFull;
        mem.slice[0] = uniforms;

        pass.setPipeline(gctx.lookupResource(self.pipeline).?);
        pass.setBindGroup(0, gctx.lookupResource(self.bind_group).?, &.{mem.offset});
        pass.draw(quad_vertex_count, self.instance_count, 0, 0);
        pass.end();
        pass.release();

        const command_buffer = encoder.finish(null);
        defer command_buffer.release();
        gctx.submit(&.{command_buffer});

        if (gctx.present() == .swap_chain_resized) {}
    }
};

pub const GaussGame = @This();

allocator: std.mem.Allocator,
scene: splat.Scene,
scene_bounds: SceneBounds,
camera: Camera,
draw_data: SplatDrawData,
uniforms: SplatUniforms,
renderer: ?SplatRenderer = null,
is_dragging: bool = false,
last_mouse_position: [2]f64 = .{ 0.0, 0.0 },
auto_rotate: bool = true,
pending_sort: bool = false,
sort_timer: f32 = 0.0,

pub fn init(
    allocator: std.mem.Allocator,
    scene: splat.Scene,
    scene_bounds: SceneBounds,
    camera: Camera,
    draw_data: SplatDrawData,
    uniforms: SplatUniforms,
) GaussGame {
    return .{
        .allocator = allocator,
        .scene = scene,
        .scene_bounds = scene_bounds,
        .camera = camera,
        .draw_data = draw_data,
        .uniforms = uniforms,
    };
}

pub fn deinit(self: *GaussGame) void {
    self.draw_data.deinit(self.allocator);
    self.scene.deinit(self.allocator);
}

pub fn initApp(self: *GaussGame, app: *zonk.App) !void {
    self.renderer = try SplatRenderer.init(self.allocator, app.graphics.gfx, self.draw_data);
    if (self.renderer) |*renderer| {
        try renderer.uploadSortedIndices(self.allocator, app.graphics.gfx, self.scene, self.camera.view());
    }
}

pub fn deinitApp(self: *GaussGame, app: *zonk.App) void {
    if (self.renderer) |*renderer| {
        renderer.deinit(app.graphics.gfx);
    }
}

pub fn update(self: *GaussGame, app: *zonk.App) !void {
    var camera_changed = false;
    var force_sort = false;
    self.sort_timer += app.delta_time;

    const scroll = app.consumeScrollDelta();
    if (scroll[1] != 0.0) {
        self.auto_rotate = false;
        zoomCamera(&self.camera, self.scene_bounds, scroll[1]);
        camera_changed = true;
        force_sort = true;
    }

    const mouse_position = app.window.getCursorPos();
    const left_down = app.window.getMouseButton(.left) == .press;
    const was_dragging = self.is_dragging;
    if (left_down) {
        if (self.is_dragging) {
            const delta_x: f32 = @floatCast(mouse_position[0] - self.last_mouse_position[0]);
            const delta_y: f32 = @floatCast(mouse_position[1] - self.last_mouse_position[1]);
            if (delta_x != 0.0 or delta_y != 0.0) {
                self.auto_rotate = false;
                orbitCamera(&self.camera, delta_x, delta_y);
                camera_changed = true;
            }
        } else {
            self.is_dragging = true;
        }
        self.last_mouse_position = mouse_position;
    } else {
        self.is_dragging = false;
        self.last_mouse_position = mouse_position;
        if (was_dragging) {
            force_sort = true;
        }
    }

    if (self.auto_rotate) {
        orbitCamera(&self.camera, 72.0 * app.delta_time, 0.0);
        camera_changed = true;
    }

    if (!camera_changed) {
        if (force_sort and self.pending_sort) {
            try self.resortAndUpload(app);
        }
        return;
    }

    self.updateUniforms();
    self.pending_sort = true;
    if (force_sort or self.sort_timer >= sort_interval_seconds) {
        try self.resortAndUpload(app);
    }
}

fn updateUniforms(self: *GaussGame) void {
    self.uniforms = makeSplatUniforms(
        self.camera,
        @as(f32, @floatFromInt(window_width)) / @as(f32, @floatFromInt(window_height)),
        @floatFromInt(window_width),
        @floatFromInt(window_height),
    );
}

fn resortAndUpload(self: *GaussGame, app: *zonk.App) !void {
    if (self.renderer) |*renderer| {
        try renderer.uploadSortedIndices(self.allocator, app.graphics.gfx, self.scene, self.camera.view());
    }
    self.pending_sort = false;
    self.sort_timer = 0.0;
}

pub fn layout(_: *GaussGame, _: usize, _: usize) zonk.Game.LayoutDim {
    return .{ .width = screen_width, .height = screen_height };
}

pub fn draw(self: *GaussGame, app: *zonk.App) !void {
    if (self.renderer) |*renderer| {
        try renderer.draw(app.graphics.gfx, self.uniforms);
    }
}

fn lessThanF32(_: void, lhs: f32, rhs: f32) bool {
    return lhs < rhs;
}

fn lessThanSortEntry(_: void, lhs: SplatSortEntry, rhs: SplatSortEntry) bool {
    return lhs.camera_z < rhs.camera_z;
}

fn makeGpuGaussian(gaussian: splat.Gaussian) GpuGaussian {
    return .{
        .centroid = gaussian.centroid,
        .opacity = gaussian.opacity,
        .scale = gaussian.scale,
        .rotation = gaussian.rotation,
    };
}

fn makeSplatUniforms(camera: Camera, aspect: f32, viewport_width: f32, viewport_height: f32) SplatUniforms {
    return .{
        .view = zmath.matToArr(camera.view()),
        .projection = zmath.matToArr(camera.projection(aspect)),
        .camera_position = .{
            camera.position[0],
            camera.position[1],
            camera.position[2],
            1.0,
        },
        .viewport_size = .{ viewport_width, viewport_height },
        .near_plane = camera.near,
    };
}

fn collectDrawData(gpa: std.mem.Allocator, scene: splat.Scene) !SplatDrawData {
    var draw_data = SplatDrawData{};
    errdefer draw_data.deinit(gpa);
    try draw_data.gaussians.ensureTotalCapacity(gpa, scene.splats.items.len);
    try draw_data.spherical_harmonics.ensureTotalCapacity(gpa, scene.splats.items.len * splat.SH_FLOAT_COUNT);

    for (scene.splats.items) |gaussian| {
        draw_data.gaussians.appendAssumeCapacity(makeGpuGaussian(gaussian));
        draw_data.spherical_harmonics.appendSliceAssumeCapacity(&gaussian.spherical_harmonics);
    }

    return draw_data;
}

fn computeSceneBounds(gpa: std.mem.Allocator, scene: splat.Scene) !SceneBounds {
    if (scene.splats.items.len == 0) {
        return .{};
    }

    var sum = zmath.f32x4s(0.0);
    for (scene.splats.items) |gaussian| {
        sum += zmath.loadArr3(gaussian.centroid);
    }

    var bounds = SceneBounds{
        .center = sum / zmath.f32x4s(@floatFromInt(scene.splats.items.len)),
    };

    const distances = try gpa.alloc(f32, scene.splats.items.len);
    defer gpa.free(distances);

    for (scene.splats.items, 0..) |gaussian, i| {
        distances[i] = zmath.length3(zmath.loadArr3(gaussian.centroid) - bounds.center)[0];
    }

    std.sort.pdq(f32, distances, {}, lessThanF32);

    const index: usize = @intFromFloat(@as(f32, @floatFromInt(distances.len - 1)) * 0.95);
    bounds.radius = @max(distances[index] * 1.5, 1.0);
    return bounds;
}

fn makeDefaultCamera(bounds: SceneBounds) Camera {
    return .{
        .position = bounds.center + zmath.f32x4(0.0, 0.0, -bounds.radius * 2.5, 0.0),
        .target = bounds.center,
        .near = bounds.radius * 0.01,
        .far = bounds.radius * 10.0,
    };
}

fn zoomCamera(camera: *Camera, bounds: SceneBounds, scroll_y: f32) void {
    const direction = zmath.normalize3(camera.position - camera.target);
    const current_distance = zmath.length3(camera.position - camera.target)[0];
    const zoom_factor = std.math.pow(f32, 0.88, scroll_y);
    const min_distance = bounds.radius * 0.25;
    const max_distance = bounds.radius * 8.0;
    const next_distance = std.math.clamp(current_distance * zoom_factor, min_distance, max_distance);
    camera.position = camera.target + direction * zmath.f32x4s(next_distance);
}

fn orbitCamera(camera: *Camera, delta_x: f32, delta_y: f32) void {
    const sensitivity = 0.006;
    const offset = camera.position - camera.target;
    const yaw = zmath.rotationY(-delta_x * sensitivity);
    const right = zmath.normalize3(zmath.cross3(zmath.normalize3(camera.target - camera.position), camera.up));
    const pitch = rotationAroundAxis(right, -delta_y * sensitivity);
    const rotated = zmath.mul(zmath.mul(offset, pitch), yaw);
    camera.position = camera.target + rotated;
}

fn rotationAroundAxis(axis: Vec3, angle: f32) zmath.Mat {
    const normalized = zmath.normalize3(axis);
    const x = normalized[0];
    const y = normalized[1];
    const z = normalized[2];
    const c = @cos(angle);
    const s = @sin(angle);
    const t = 1.0 - c;

    return .{
        zmath.f32x4(t * x * x + c, t * x * y + s * z, t * x * z - s * y, 0.0),
        zmath.f32x4(t * x * y - s * z, t * y * y + c, t * y * z + s * x, 0.0),
        zmath.f32x4(t * x * z + s * y, t * y * z - s * x, t * z * z + c, 0.0),
        zmath.f32x4(0.0, 0.0, 0.0, 1.0),
    };
}

fn printCameraVisibility(scene: splat.Scene, camera: Camera) void {
    var visible_count: usize = 0;
    var min_z: f32 = std.math.floatMax(f32);
    var max_z: f32 = -std.math.floatMax(f32);
    const view = camera.view();
    for (scene.splats.items) |gaussian| {
        const camera_centroid = zmath.mul(zmath.loadArr3w(gaussian.centroid, 1.0), view);
        min_z = @min(min_z, camera_centroid[2]);
        max_z = @max(max_z, camera_centroid[2]);
        if (camera_centroid[2] < -camera.near) {
            visible_count += 1;
        }
    }
    std.debug.print("camera visibility: {d}/{d} splats past near plane, camera z range=({d:.4}, {d:.4})\n", .{
        visible_count,
        scene.splats.items.len,
        min_z,
        max_z,
    });
}

pub fn main(process_init: std.process.Init) !void {
    const gpa = process_init.gpa;
    const io = process_init.io;

    var args = try std.process.Args.Iterator.initAllocator(process_init.minimal.args, gpa);
    defer args.deinit();
    _ = args.skip();

    const file_path = args.next() orelse {
        std.debug.print("no input file specified\n", .{});
        return;
    };
    std.debug.print("input file: {s}\n", .{file_path});

    var scene = splat.loadPly(io, gpa, file_path) catch |err| {
        std.debug.print("failed: {any}\n", .{err});
        return;
    };
    var scene_owned_by_main = true;
    errdefer if (scene_owned_by_main) scene.deinit(gpa);
    for (scene.splats.items) |*gaussian| {
        gaussian.centroid[1] = -gaussian.centroid[1];
    }

    const bounds = try computeSceneBounds(gpa, scene);
    std.debug.print("scene bounds: center=({d:.4}, {d:.4}, {d:.4}), radius={d:.4}\n", .{
        bounds.center[0],
        bounds.center[1],
        bounds.center[2],
        bounds.radius,
    });

    const camera = makeDefaultCamera(bounds);
    printCameraVisibility(scene, camera);
    const viewport_width: f32 = @floatFromInt(window_width);
    const viewport_height: f32 = @floatFromInt(window_height);
    const aspect = viewport_width / viewport_height;
    const uniforms = makeSplatUniforms(camera, aspect, viewport_width, viewport_height);
    const draw_data = try collectDrawData(gpa, scene);
    std.debug.print("camera: position=({d:.4}, {d:.4}, {d:.4}), target=({d:.4}, {d:.4}, {d:.4}), near={d:.4}, far={d:.4}\n", .{
        camera.position[0],
        camera.position[1],
        camera.position[2],
        camera.target[0],
        camera.target[1],
        camera.target[2],
        camera.near,
        camera.far,
    });
    std.debug.print("splat draw data: quad vertices={d}, gaussian instances={d}, sh floats={d}, uniforms={d} bytes\n", .{
        quad_vertex_count,
        draw_data.gaussians.items.len,
        draw_data.spherical_harmonics.items.len,
        @sizeOf(SplatUniforms),
    });
    var game = GaussGame.init(gpa, scene, bounds, camera, draw_data, uniforms);
    scene_owned_by_main = false;
    defer game.deinit();

    const config = zonk.GameConfig{
        .title = "Gauss Splats",
        .width = window_width,
        .height = window_height,
        .vsync = true,
        .uncapped_fps = false,
    };

    try zonk.runApp(GaussGame, &game, gpa, io, config);
}
