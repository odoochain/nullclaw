const std = @import("std");
const bootstrap_mod = @import("../bootstrap/root.zig");
const isPathSafe = @import("path_security.zig").isPathSafe;

pub const PrepareWorkspacePathError = error{
    OutOfMemory,
    AbsolutePathsNotAllowed,
    PathContainsNullBytes,
    UnsafePath,
};

pub const WorkspacePathInfo = struct {
    full_path: []u8,
    workspace_resolved: ?[]u8,

    pub fn deinit(self: WorkspacePathInfo, allocator: std.mem.Allocator) void {
        allocator.free(self.full_path);
        if (self.workspace_resolved) |resolved| allocator.free(resolved);
    }

    pub fn workspacePath(self: WorkspacePathInfo) []const u8 {
        return self.workspace_resolved orelse "";
    }
};

pub fn prepareWorkspacePath(
    allocator: std.mem.Allocator,
    workspace_dir: []const u8,
    path: []const u8,
    allow_absolute_paths: bool,
) PrepareWorkspacePathError!WorkspacePathInfo {
    const full_path = if (std.fs.path.isAbsolute(path)) blk: {
        if (!allow_absolute_paths) return error.AbsolutePathsNotAllowed;
        if (std.mem.indexOfScalar(u8, path, 0) != null) return error.PathContainsNullBytes;
        break :blk try allocator.dupe(u8, path);
    } else blk: {
        if (!isPathSafe(path)) return error.UnsafePath;
        break :blk try std.fs.path.join(allocator, &.{ workspace_dir, path });
    };
    errdefer allocator.free(full_path);

    const workspace_resolved = std.fs.cwd().realpathAlloc(allocator, workspace_dir) catch null;
    errdefer if (workspace_resolved) |resolved| allocator.free(resolved);

    return .{
        .full_path = full_path,
        .workspace_resolved = workspace_resolved,
    };
}

pub fn resolveNearestExistingAncestor(allocator: std.mem.Allocator, path: []const u8) ![]const u8 {
    return std.fs.cwd().realpathAlloc(allocator, path) catch |err| switch (err) {
        error.FileNotFound => {
            const parent = std.fs.path.dirname(path) orelse return err;
            if (std.mem.eql(u8, parent, path)) return err;
            return resolveNearestExistingAncestor(allocator, parent);
        },
        else => return err,
    };
}

pub fn bootstrapRootFilename(path: []const u8) ?[]const u8 {
    if (std.fs.path.isAbsolute(path)) return null;
    const basename = std.fs.path.basename(path);
    if (!std.mem.eql(u8, basename, path)) return null;
    if (!bootstrap_mod.isBootstrapFilename(basename)) return null;
    return basename;
}

test "bootstrapRootFilename returns basename for workspace root bootstrap file" {
    try std.testing.expectEqualStrings("BOOTSTRAP.md", bootstrapRootFilename("BOOTSTRAP.md").?);
}

test "prepareWorkspacePath joins relative path and resolves workspace" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const workspace_dir = try tmp.dir.realpathAlloc(std.testing.allocator, ".");
    defer std.testing.allocator.free(workspace_dir);

    const info = try prepareWorkspacePath(std.testing.allocator, workspace_dir, "notes/todo.md", false);
    defer info.deinit(std.testing.allocator);

    const expected = try std.fs.path.join(std.testing.allocator, &.{ workspace_dir, "notes/todo.md" });
    defer std.testing.allocator.free(expected);

    try std.testing.expectEqualStrings(expected, info.full_path);
    try std.testing.expectEqualStrings(workspace_dir, info.workspacePath());
}

test "prepareWorkspacePath rejects absolute path when not allowed" {
    const absolute_path = if (std.fs.path.sep == '\\')
        "C:\\workspace\\todo.md"
    else
        "/workspace/todo.md";

    try std.testing.expectError(
        error.AbsolutePathsNotAllowed,
        prepareWorkspacePath(std.testing.allocator, ".", absolute_path, false),
    );
}

test "bootstrapRootFilename rejects nested and absolute paths" {
    try std.testing.expect(bootstrapRootFilename("docs/BOOTSTRAP.md") == null);

    const absolute_path = if (std.fs.path.sep == '\\')
        "C:\\workspace\\BOOTSTRAP.md"
    else
        "/workspace/BOOTSTRAP.md";
    try std.testing.expect(bootstrapRootFilename(absolute_path) == null);
}

test "resolveNearestExistingAncestor returns nearest existing parent" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    try tmp.dir.makePath("existing/child");

    const existing_path = try tmp.dir.realpathAlloc(std.testing.allocator, "existing/child");
    defer std.testing.allocator.free(existing_path);

    const missing_path = try std.fs.path.join(std.testing.allocator, &.{ existing_path, "missing", "leaf.txt" });
    defer std.testing.allocator.free(missing_path);

    const resolved = try resolveNearestExistingAncestor(std.testing.allocator, missing_path);
    defer std.testing.allocator.free(resolved);

    try std.testing.expectEqualStrings(existing_path, resolved);
}
