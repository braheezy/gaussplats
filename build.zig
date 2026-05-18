const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const zonk_mod = b.dependency("zonk", .{
        .max_buffer_size = b.option(u64, "max_buffer_size", "Required WebGPU max_buffer_size limit") orelse 512 * 1024 * 1024,
    }).module("zonk");
    const zmath_mod = b.dependency("zmath", .{}).module("root");

    const main_mod = b.createModule(.{
        .root_source_file = b.path("src/main.zig"),
        .target = target,
        .optimize = optimize,
        .unwind_tables = if (target.result.os.tag == .emscripten) .none else null,
    });
    main_mod.addImport("zonk", zonk_mod);
    main_mod.addImport("zmath", zmath_mod);

    const main_exe = b.addExecutable(.{
        .root_module = main_mod,
        .name = "guassplats",
    });
    b.installArtifact(main_exe);

    const run_exe = b.addRunArtifact(main_exe);
    if (b.args) |args| {
        run_exe.addArgs(args);
    }

    const run_step = b.step("run", "Run the program");
    run_step.dependOn(&run_exe.step);
}
