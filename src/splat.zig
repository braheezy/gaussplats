const std = @import("std");

pub const SH_COUNT = 16;
pub const SH_CHANNEL_COUNT = 3;
pub const SH_FLOAT_COUNT = SH_COUNT * SH_CHANNEL_COUNT;
pub const SH_REST_FLOAT_COUNT = SH_FLOAT_COUNT - SH_CHANNEL_COUNT;

pub const Gaussian = struct {
    centroid: [3]f32,
    opacity: f32,
    spherical_harmonics: [SH_FLOAT_COUNT]f32,
    scale: [3]f32,
    rotation: [4]f32,
};

const Property = struct {
    name: []const u8,
    kind: []const u8,
    offset: usize,
    size: usize,
};

pub const Scene = struct {
    splats: std.ArrayList(Gaussian) = .empty,

    pub fn deinit(self: *Scene, gpa: std.mem.Allocator) void {
        self.splats.deinit(gpa);
    }
};

fn scalarSize(kind: []const u8) !usize {
    if (std.mem.eql(u8, kind, "char") or std.mem.eql(u8, kind, "uchar") or
        std.mem.eql(u8, kind, "int8") or std.mem.eql(u8, kind, "uint8"))
    {
        return 1;
    }

    if (std.mem.eql(u8, kind, "short") or std.mem.eql(u8, kind, "ushort") or
        std.mem.eql(u8, kind, "int16") or std.mem.eql(u8, kind, "uint16"))
    {
        return 2;
    }

    if (std.mem.eql(u8, kind, "int") or std.mem.eql(u8, kind, "uint") or
        std.mem.eql(u8, kind, "float") or std.mem.eql(u8, kind, "int32") or
        std.mem.eql(u8, kind, "uint32") or std.mem.eql(u8, kind, "float32"))
    {
        return 4;
    }

    if (std.mem.eql(u8, kind, "double") or std.mem.eql(u8, kind, "float64")) {
        return 8;
    }

    return error.UnsupportedPlyScalarType;
}

fn requireProperty(properties: []const Property, name: []const u8) !Property {
    return findProperty(properties, name) orelse error.MissingPlyProperty;
}

fn findProperty(properties: []const Property, name: []const u8) ?Property {
    for (properties) |property| {
        if (std.mem.eql(u8, property.name, name)) {
            return property;
        }
    }
    return null;
}

fn readAsFloat(row: []const u8, property: Property) !f32 {
    if (property.offset + property.size > row.len) {
        return error.MalformedPlyRow;
    }

    const bytes = row[property.offset..][0..property.size];
    if (std.mem.eql(u8, property.kind, "float") or std.mem.eql(u8, property.kind, "float32")) {
        return @as(f32, @bitCast(std.mem.readInt(u32, bytes[0..4], .little)));
    }
    if (std.mem.eql(u8, property.kind, "double") or std.mem.eql(u8, property.kind, "float64")) {
        return @floatCast(@as(f64, @bitCast(std.mem.readInt(u64, bytes[0..8], .little))));
    }
    if (std.mem.eql(u8, property.kind, "char") or std.mem.eql(u8, property.kind, "int8")) {
        return @floatFromInt(@as(i8, @bitCast(bytes[0])));
    }
    if (std.mem.eql(u8, property.kind, "uchar") or std.mem.eql(u8, property.kind, "uint8")) {
        return @floatFromInt(bytes[0]);
    }
    if (std.mem.eql(u8, property.kind, "short") or std.mem.eql(u8, property.kind, "int16")) {
        return @floatFromInt(std.mem.readInt(i16, bytes[0..2], .little));
    }
    if (std.mem.eql(u8, property.kind, "ushort") or std.mem.eql(u8, property.kind, "uint16")) {
        return @floatFromInt(std.mem.readInt(u16, bytes[0..2], .little));
    }
    if (std.mem.eql(u8, property.kind, "int") or std.mem.eql(u8, property.kind, "int32")) {
        return @floatFromInt(std.mem.readInt(i32, bytes[0..4], .little));
    }
    if (std.mem.eql(u8, property.kind, "uint") or std.mem.eql(u8, property.kind, "uint32")) {
        return @floatFromInt(std.mem.readInt(u32, bytes[0..4], .little));
    }

    return error.UnsupportedPlyScalarType;
}

pub fn loadPly(io: std.Io, gpa: std.mem.Allocator, filepath: []const u8) !Scene {
    const file = try std.Io.Dir.cwd().openFile(io, filepath, .{});
    defer file.close(io);

    var buf: [1024]u8 = undefined;
    var file_reader = file.reader(io, &buf);
    const reader = &file_reader.interface;

    const raw_first_line = try reader.takeDelimiter('\n') orelse return error.InvalidFile;
    const first_line = std.mem.trimEnd(u8, raw_first_line, "\r");
    if (!std.mem.eql(u8, first_line, "ply")) {
        return error.InvalidFile;
    }

    var saw_format = false;
    var in_vertex = false;
    var vertex_count: usize = 0;
    var row_stride: usize = 0;
    var properties: std.ArrayList(Property) = .empty;
    defer {
        for (properties.items) |property| {
            gpa.free(property.name);
            gpa.free(property.kind);
        }
        properties.deinit(gpa);
    }

    while (try reader.takeDelimiter('\n')) |raw_line| {
        const line = std.mem.trimEnd(u8, raw_line, "\r");

        if (std.mem.eql(u8, line, "end_header")) {
            break;
        }

        var words = std.mem.tokenizeAny(u8, line, " \t");
        const tag = words.next() orelse continue;
        if (std.mem.eql(u8, tag, "comment")) {
            continue;
        }

        if (std.mem.eql(u8, tag, "format")) {
            const format = words.next() orelse return error.MalformedPlyHeader;
            _ = words.next() orelse return error.MalformedPlyHeader;
            if (!std.mem.eql(u8, format, "binary_little_endian")) {
                return error.UnsupportedPlyFormat;
            }
            saw_format = true;
            continue;
        }

        if (std.mem.eql(u8, tag, "element")) {
            const element_name = words.next() orelse return error.MalformedPlyHeader;
            const count_text = words.next() orelse return error.MalformedPlyHeader;
            in_vertex = std.mem.eql(u8, element_name, "vertex");
            if (in_vertex) {
                vertex_count = try std.fmt.parseInt(usize, count_text, 10);
            }
            continue;
        }

        if (in_vertex and std.mem.eql(u8, tag, "property")) {
            const property_kind = words.next() orelse return error.MalformedPlyHeader;
            const property_name = words.next() orelse return error.MalformedPlyHeader;
            if (std.mem.eql(u8, property_kind, "list")) {
                return error.UnsupportedPlyListProperty;
            }

            const size = try scalarSize(property_kind);
            try properties.append(gpa, .{
                .name = try gpa.dupe(u8, property_name),
                .kind = try gpa.dupe(u8, property_kind),
                .offset = row_stride,
                .size = size,
            });
            row_stride += size;
        }
    }

    if (!saw_format) {
        return error.MissingPlyFormat;
    }
    if (vertex_count == 0) {
        return error.MissingPlyVertices;
    }

    const x = try requireProperty(properties.items, "x");
    const y = try requireProperty(properties.items, "y");
    const z = try requireProperty(properties.items, "z");
    const dc_r = try requireProperty(properties.items, "f_dc_0");
    const dc_g = try requireProperty(properties.items, "f_dc_1");
    const dc_b = try requireProperty(properties.items, "f_dc_2");
    const opacity = try requireProperty(properties.items, "opacity");
    const scale_0 = try requireProperty(properties.items, "scale_0");
    const scale_1 = try requireProperty(properties.items, "scale_1");
    const scale_2 = try requireProperty(properties.items, "scale_2");
    const rot_0 = try requireProperty(properties.items, "rot_0");
    const rot_1 = try requireProperty(properties.items, "rot_1");
    const rot_2 = try requireProperty(properties.items, "rot_2");
    const rot_3 = try requireProperty(properties.items, "rot_3");

    var sh_rest: std.ArrayList(Property) = .empty;
    defer sh_rest.deinit(gpa);
    for (0..SH_REST_FLOAT_COUNT) |i| {
        var name_buf: [32]u8 = undefined;
        const name = try std.fmt.bufPrint(&name_buf, "f_rest_{d}", .{i});
        const property = findProperty(properties.items, name) orelse break;
        try sh_rest.append(gpa, property);
    }

    var extra_sh_name_buf: [32]u8 = undefined;
    const extra_sh_name = try std.fmt.bufPrint(&extra_sh_name_buf, "f_rest_{d}", .{SH_REST_FLOAT_COUNT});
    if (findProperty(properties.items, extra_sh_name) != null) {
        return error.TooManySphericalHarmonicCoefficients;
    }
    if (sh_rest.items.len % SH_CHANNEL_COUNT != 0) {
        return error.InvalidSphericalHarmonicChannelCount;
    }
    const rest_count_per_channel = sh_rest.items.len / SH_CHANNEL_COUNT;

    std.debug.print("vertex count: {d}, row stride: {d}, properties: {d}\n", .{
        vertex_count,
        row_stride,
        properties.items.len,
    });
    std.debug.print("required offsets: x={d}, y={d}, z={d}, opacity={d}, scale=({d},{d},{d}), rot=({d},{d},{d},{d}), dc=({d},{d},{d})\n", .{
        x.offset,
        y.offset,
        z.offset,
        opacity.offset,
        scale_0.offset,
        scale_1.offset,
        scale_2.offset,
        rot_0.offset,
        rot_1.offset,
        rot_2.offset,
        rot_3.offset,
        dc_r.offset,
        dc_g.offset,
        dc_b.offset,
    });
    std.debug.print("spherical harmonic rest: {d} total, {d} per channel\n", .{
        sh_rest.items.len,
        rest_count_per_channel,
    });

    var result = Scene{};
    errdefer result.splats.deinit(gpa);
    try result.splats.ensureTotalCapacity(gpa, vertex_count);

    const row = try gpa.alloc(u8, row_stride);
    defer gpa.free(row);

    for (0..vertex_count) |_| {
        reader.readSliceAll(row) catch |err| switch (err) {
            error.EndOfStream => return error.PlyEndedBeforeAllVertexRecords,
            else => |e| return e,
        };

        var splat = Gaussian{
            .centroid = .{
                try readAsFloat(row, x),
                try readAsFloat(row, y),
                try readAsFloat(row, z),
            },
            .opacity = try readAsFloat(row, opacity),
            .spherical_harmonics = [_]f32{0.0} ** SH_FLOAT_COUNT,
            .scale = .{
                try readAsFloat(row, scale_0),
                try readAsFloat(row, scale_1),
                try readAsFloat(row, scale_2),
            },
            .rotation = .{
                try readAsFloat(row, rot_0),
                try readAsFloat(row, rot_1),
                try readAsFloat(row, rot_2),
                try readAsFloat(row, rot_3),
            },
        };

        splat.spherical_harmonics[0] = try readAsFloat(row, dc_r);
        splat.spherical_harmonics[1] = try readAsFloat(row, dc_g);
        splat.spherical_harmonics[2] = try readAsFloat(row, dc_b);

        for (0..SH_CHANNEL_COUNT) |channel| {
            for (0..rest_count_per_channel) |coeff| {
                const ply_rest_index = channel * rest_count_per_channel + coeff;
                const splat_coeff = coeff + 1;
                const splat_index = splat_coeff * SH_CHANNEL_COUNT + channel;
                splat.spherical_harmonics[splat_index] =
                    try readAsFloat(row, sh_rest.items[ply_rest_index]);
            }
        }

        try result.splats.append(gpa, splat);
    }

    return result;
}
