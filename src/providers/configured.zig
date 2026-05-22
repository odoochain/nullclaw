//! Helpers for constructing provider holders from resolved runtime config.

const std = @import("std");
const Config = @import("../config.zig").Config;
const config_types = @import("../config_types.zig");
const compatible = @import("compatible.zig");
const factory = @import("factory.zig");

pub fn holderFromConfig(
    allocator: std.mem.Allocator,
    cfg: *const Config,
    provider_name: []const u8,
    api_key: ?[]const u8,
) factory.ProviderHolder {
    return factory.ProviderHolder.fromConfigWithApiMode(
        allocator,
        provider_name,
        api_key,
        cfg.getProviderBaseUrl(provider_name),
        cfg.getProviderNativeTools(provider_name),
        cfg.getProviderUserAgent(provider_name),
        cfg.getProviderApiMode(provider_name),
        cfg.getProviderMaxStreamingPromptBytes(provider_name),
        cfg.getProviderChatTemplateEnableThinkingParam(provider_name),
        cfg.getProviderExtraBodyParams(provider_name),
    );
}

pub fn holderFromEntry(
    allocator: std.mem.Allocator,
    provider_name: []const u8,
    api_key: ?[]const u8,
    entry: ?config_types.ProviderEntry,
) factory.ProviderHolder {
    return factory.ProviderHolder.fromConfigWithApiMode(
        allocator,
        provider_name,
        api_key,
        if (entry) |e| e.base_url else null,
        if (entry) |e| e.native_tools else true,
        if (entry) |e| e.user_agent else null,
        if (entry) |e| e.api_mode else .chat_completions,
        if (entry) |e| e.max_streaming_prompt_bytes else null,
        if (entry) |e| e.chat_template_enable_thinking_param else false,
        if (entry) |e| e.extra_body_params else null,
    );
}

test "holderFromConfig applies provider runtime settings" {
    const allocator = std.testing.allocator;
    const providers_cfg = [_]config_types.ProviderEntry{.{
        .name = "custom-local",
        .base_url = "http://localhost:4321/v1",
        .native_tools = false,
        .user_agent = "nullclaw-test",
        .api_mode = .responses,
        .chat_template_enable_thinking_param = true,
        .max_streaming_prompt_bytes = 123,
        .extra_body_params = "{\"seed\":1}",
    }};
    const cfg = Config{
        .workspace_dir = "/tmp/nullclaw-test",
        .config_path = "/tmp/nullclaw-test/config.json",
        .allocator = allocator,
        .providers = &providers_cfg,
    };

    var holder = holderFromConfig(allocator, &cfg, "custom-local", null);
    defer holder.deinit();

    switch (holder) {
        .compatible => |*provider| {
            try std.testing.expectEqualStrings("http://localhost:4321/v1", provider.base_url);
            try std.testing.expect(!provider.native_tools);
            try std.testing.expectEqualStrings("nullclaw-test", provider.user_agent.?);
            try std.testing.expectEqual(compatible.CompatibleApiMode.responses, provider.api_mode);
            try std.testing.expect(provider.chat_template_enable_thinking_param);
            try std.testing.expectEqual(@as(?usize, 123), provider.max_streaming_prompt_bytes);
            try std.testing.expectEqualStrings("{\"seed\":1}", provider.extra_body_params.?);
        },
        else => return error.TestUnexpectedResult,
    }
}
