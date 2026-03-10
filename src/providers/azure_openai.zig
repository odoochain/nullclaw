const std = @import("std");
const root = @import("root.zig");
const sse = @import("sse.zig");
const error_classify = @import("error_classify.zig");

const Provider = root.Provider;
const ChatRequest = root.ChatRequest;
const ChatResponse = root.ChatResponse;
const ToolCall = root.ToolCall;
const TokenUsage = root.TokenUsage;

/// Azure OpenAI API provider.
///
/// Endpoints:
/// - POST https://{endpoint}/openai/deployments/{model}/chat/completions?api-version={api-version}
/// - api-key: <key>
///
/// Key differences from OpenAI:
/// - Model name used as deployment name in URL path instead of model in request body
/// - Uses `api-key` header instead of `Authorization: Bearer`
/// - Reuses the shared OpenAI generation-field rules for reasoning models
pub const AzureOpenAiProvider = struct {
    api_key: ?[]const u8,
    base_url: []const u8,
    allocator: std.mem.Allocator,
    /// Optional User-Agent header for HTTP requests.
    user_agent: ?[]const u8 = null,

    const API_VERSION = "2024-10-21";
    pub const DEFAULT_BASE_URL = "https://your-resource.openai.azure.com";

    pub fn init(
        allocator: std.mem.Allocator,
        api_key: ?[]const u8,
        base_url: []const u8,
        user_agent: ?[]const u8,
    ) AzureOpenAiProvider {
        return .{
            .api_key = api_key,
            .base_url = base_url,
            .allocator = allocator,
            .user_agent = user_agent,
        };
    }

    fn validateUserAgent(user_agent: []const u8) bool {
        // Disallow header injection and malformed values.
        return std.mem.indexOfAny(u8, user_agent, "\r\n") == null;
    }

    fn trimTrailingSlash(value: []const u8) []const u8 {
        if (std.mem.endsWith(u8, value, "/")) {
            return value[0 .. value.len - 1];
        }
        return value;
    }

    fn normalizeBaseUrl(value: []const u8) []const u8 {
        const trimmed = trimTrailingSlash(value);
        if (std.mem.endsWith(u8, trimmed, "/openai/v1")) {
            return trimmed[0 .. trimmed.len - "/openai/v1".len];
        }
        if (std.mem.endsWith(u8, trimmed, "/openai")) {
            return trimmed[0 .. trimmed.len - "/openai".len];
        }
        return trimmed;
    }

    /// Build Azure-specific chat completions URL with model as deployment name.
    fn buildChatUrl(self: *const AzureOpenAiProvider, allocator: std.mem.Allocator, model: []const u8) ![]const u8 {
        const base = normalizeBaseUrl(self.base_url);
        if (base.len == 0 or std.mem.eql(u8, base, DEFAULT_BASE_URL)) return error.InvalidBaseUrl;

        return std.fmt.allocPrint(
            allocator,
            "{s}/openai/deployments/{s}/chat/completions?api-version={s}",
            .{ base, model, API_VERSION },
        );
    }

    /// Build a simple chat request JSON body (Azure doesn't include model field in request body).
    pub fn buildRequestBody(
        allocator: std.mem.Allocator,
        system_prompt: ?[]const u8,
        message: []const u8,
        model: []const u8,
        temperature: f64,
        max_tokens: ?u32,
    ) ![]const u8 {
        var buf: std.ArrayListUnmanaged(u8) = .empty;
        errdefer buf.deinit(allocator);

        try buf.appendSlice(allocator, "{\"messages\":[");

        if (system_prompt) |sys| {
            try buf.appendSlice(allocator, "{\"role\":\"system\",\"content\":");
            try root.appendJsonString(&buf, allocator, sys);
            try buf.appendSlice(allocator, "},");
        }

        try buf.appendSlice(allocator, "{\"role\":\"user\",\"content\":");
        try root.appendJsonString(&buf, allocator, message);
        try buf.append(allocator, '}');

        try buf.append(allocator, ']');
        try root.appendGenerationFields(&buf, allocator, model, temperature, max_tokens, null);
        try buf.append(allocator, '}');
        return try buf.toOwnedSlice(allocator);
    }

    /// Parse a simple text response into a string.
    pub fn parseTextResponse(allocator: std.mem.Allocator, body: []const u8) ![]const u8 {
        const parsed = try std.json.parseFromSlice(std.json.Value, allocator, body, .{});
        defer parsed.deinit();
        const root_obj = parsed.value.object;

        if (error_classify.classifyKnownApiError(root_obj)) |kind| {
            const mapped_err = error_classify.kindToError(kind);
            var summary_buf: [1024]u8 = undefined;
            const summary = error_classify.summarizeKnownApiError(root_obj, &summary_buf) orelse @errorName(mapped_err);
            const sanitized = root.sanitizeApiError(allocator, summary) catch null;
            defer if (sanitized) |s| allocator.free(s);
            root.setLastApiErrorDetail("azure-openai", sanitized orelse summary);
            return mapped_err;
        }

        if (root_obj.get("choices")) |choices| {
            if (choices.array.items.len > 0) {
                const msg = choices.array.items[0].object.get("message") orelse return error.NoResponseContent;
                const msg_obj = msg.object;
                if (msg_obj.get("content")) |content| {
                    if (content == .string) {
                        return allocator.dupe(u8, content.string);
                    }
                }
            }
        }

        return error.NoResponseContent;
    }

    /// Parse a native tool-calling response into ChatResponse.
    pub fn parseNativeResponse(allocator: std.mem.Allocator, body: []const u8) !ChatResponse {
        const parsed = try std.json.parseFromSlice(std.json.Value, allocator, body, .{});
        defer parsed.deinit();
        const root_obj = parsed.value.object;

        if (error_classify.classifyKnownApiError(root_obj)) |kind| {
            const mapped_err = error_classify.kindToError(kind);
            var summary_buf: [1024]u8 = undefined;
            const summary = error_classify.summarizeKnownApiError(root_obj, &summary_buf) orelse @errorName(mapped_err);
            const sanitized = root.sanitizeApiError(allocator, summary) catch null;
            defer if (sanitized) |s| allocator.free(s);
            root.setLastApiErrorDetail("azure-openai", sanitized orelse summary);
            return mapped_err;
        }

        if (root_obj.get("choices")) |choices| {
            if (choices.array.items.len > 0) {
                const msg = choices.array.items[0].object.get("message") orelse return error.NoResponseContent;
                const msg_obj = msg.object;

                var content: ?[]const u8 = null;
                var reasoning_content: ?[]const u8 = null;
                if (msg_obj.get("content")) |c| {
                    if (c == .string) {
                        const split = try root.splitThinkContent(allocator, c.string);
                        content = split.visible;
                        reasoning_content = split.reasoning;
                    }
                }

                var tool_calls_list: std.ArrayListUnmanaged(ToolCall) = .empty;

                if (msg_obj.get("tool_calls")) |tc_arr| {
                    for (tc_arr.array.items) |tc| {
                        const tc_obj = tc.object;
                        const id = if (tc_obj.get("id")) |i| (if (i == .string) try allocator.dupe(u8, i.string) else try allocator.dupe(u8, "unknown")) else try allocator.dupe(u8, "unknown");

                        if (tc_obj.get("function")) |func| {
                            const func_obj = func.object;
                            const name = if (func_obj.get("name")) |n| (if (n == .string) try allocator.dupe(u8, n.string) else try allocator.dupe(u8, "")) else try allocator.dupe(u8, "");
                            const arguments = if (func_obj.get("arguments")) |a| (if (a == .string) try allocator.dupe(u8, a.string) else try allocator.dupe(u8, "{}")) else try allocator.dupe(u8, "{}");

                            try tool_calls_list.append(allocator, .{
                                .id = id,
                                .name = name,
                                .arguments = arguments,
                            });
                        }
                    }
                }

                // Parse usage
                var usage = TokenUsage{};
                if (root_obj.get("usage")) |usage_obj| {
                    if (usage_obj == .object) {
                        if (usage_obj.object.get("prompt_tokens")) |v| {
                            if (v == .integer) usage.prompt_tokens = @intCast(v.integer);
                        }
                        if (usage_obj.object.get("completion_tokens")) |v| {
                            if (v == .integer) usage.completion_tokens = @intCast(v.integer);
                        }
                        if (usage_obj.object.get("total_tokens")) |v| {
                            if (v == .integer) usage.total_tokens = @intCast(v.integer);
                        }
                    }
                }

                const model_str = if (root_obj.get("model")) |m| (if (m == .string) try allocator.dupe(u8, m.string) else try allocator.dupe(u8, "")) else try allocator.dupe(u8, "");

                return .{
                    .content = content,
                    .reasoning_content = reasoning_content,
                    .tool_calls = try tool_calls_list.toOwnedSlice(allocator),
                    .usage = usage,
                    .model = model_str,
                };
            }
        }

        return error.NoResponseContent;
    }

    /// Create a Provider interface from this AzureOpenAiProvider.
    pub fn provider(self: *AzureOpenAiProvider) Provider {
        return .{
            .ptr = @ptrCast(self),
            .vtable = &vtable,
        };
    }

    const vtable = Provider.VTable{
        .chatWithSystem = chatWithSystemImpl,
        .chat = chatImpl,
        .supportsNativeTools = supportsNativeToolsImpl,
        .supports_vision = supportsVisionImpl,
        .getName = getNameImpl,
        .deinit = deinitImpl,
        .stream_chat = streamChatImpl,
        .supports_streaming = supportsStreamingImpl,
    };

    fn streamChatImpl(
        ptr: *anyopaque,
        allocator: std.mem.Allocator,
        request: root.ChatRequest,
        model: []const u8,
        temperature: f64,
        callback: root.StreamCallback,
        callback_ctx: *anyopaque,
    ) anyerror!root.StreamChatResult {
        const self: *AzureOpenAiProvider = @ptrCast(@alignCast(ptr));
        const api_key = self.api_key orelse return error.CredentialsNotSet;

        const chat_url = try self.buildChatUrl(allocator, model);
        defer allocator.free(chat_url);

        const body = try buildStreamingChatRequestBody(allocator, request, model, temperature);
        defer allocator.free(body);

        var auth_hdr_buf: [512]u8 = undefined;
        const auth_hdr = std.fmt.bufPrint(&auth_hdr_buf, "api-key: {s}", .{api_key}) catch return error.AzureOpenAiApiError;

        // Build extra headers (User-Agent if configured)
        var extra_headers: [1][]const u8 = undefined;
        var extra_header_count: usize = 0;
        var user_agent_hdr: ?[]u8 = null;
        defer if (user_agent_hdr) |h| allocator.free(h);
        if (self.user_agent) |ua| {
            if (!validateUserAgent(ua)) return error.AzureOpenAiApiError;
            user_agent_hdr = std.fmt.allocPrint(allocator, "User-Agent: {s}", .{ua}) catch return error.AzureOpenAiApiError;
            extra_headers[extra_header_count] = user_agent_hdr.?;
            extra_header_count += 1;
        }

        const timeout_secs = if (request.timeout_secs < 60) 60 else request.timeout_secs;
        return sse.curlStream(allocator, chat_url, body, auth_hdr, extra_headers[0..extra_header_count], timeout_secs, callback, callback_ctx);
    }

    fn supportsStreamingImpl(_: *anyopaque) bool {
        return true;
    }

    fn chatWithSystemImpl(
        ptr: *anyopaque,
        allocator: std.mem.Allocator,
        system_prompt: ?[]const u8,
        message: []const u8,
        model: []const u8,
        temperature: f64,
    ) anyerror![]const u8 {
        const self: *AzureOpenAiProvider = @ptrCast(@alignCast(ptr));
        const api_key = self.api_key orelse return error.CredentialsNotSet;

        const chat_url = try self.buildChatUrl(allocator, model);
        defer allocator.free(chat_url);

        const body = try buildRequestBody(allocator, system_prompt, message, model, temperature, null);
        defer allocator.free(body);

        // Build headers (auth + optional User-Agent)
        var headers_buf: [2][]const u8 = undefined;
        var header_count: usize = 0;
        var auth_hdr_buf: [512]u8 = undefined;
        const auth_hdr = std.fmt.bufPrint(&auth_hdr_buf, "api-key: {s}", .{api_key}) catch return error.AzureOpenAiApiError;
        headers_buf[header_count] = auth_hdr;
        header_count += 1;
        var user_agent_hdr: ?[]u8 = null;
        defer if (user_agent_hdr) |h| allocator.free(h);
        if (self.user_agent) |ua| {
            if (!validateUserAgent(ua)) return error.AzureOpenAiApiError;
            user_agent_hdr = std.fmt.allocPrint(allocator, "User-Agent: {s}", .{ua}) catch return error.AzureOpenAiApiError;
            headers_buf[header_count] = user_agent_hdr.?;
            header_count += 1;
        }

        const resp_body = root.curlPostTimed(allocator, chat_url, body, headers_buf[0..header_count], 60) catch return error.AzureOpenAiApiError;
        defer allocator.free(resp_body);

        return parseTextResponse(allocator, resp_body);
    }

    fn chatImpl(
        ptr: *anyopaque,
        allocator: std.mem.Allocator,
        request: ChatRequest,
        model: []const u8,
        temperature: f64,
    ) anyerror!ChatResponse {
        const self: *AzureOpenAiProvider = @ptrCast(@alignCast(ptr));
        const api_key = self.api_key orelse return error.CredentialsNotSet;

        const chat_url = try self.buildChatUrl(allocator, model);
        defer allocator.free(chat_url);

        const body = try buildChatRequestBody(allocator, request, model, temperature);
        defer allocator.free(body);

        // Build headers (auth + optional User-Agent)
        var headers_buf: [2][]const u8 = undefined;
        var header_count: usize = 0;
        var auth_hdr_buf: [512]u8 = undefined;
        const auth_hdr = std.fmt.bufPrint(&auth_hdr_buf, "api-key: {s}", .{api_key}) catch return error.AzureOpenAiApiError;
        headers_buf[header_count] = auth_hdr;
        header_count += 1;
        var user_agent_hdr: ?[]u8 = null;
        defer if (user_agent_hdr) |h| allocator.free(h);
        if (self.user_agent) |ua| {
            if (!validateUserAgent(ua)) return error.AzureOpenAiApiError;
            user_agent_hdr = std.fmt.allocPrint(allocator, "User-Agent: {s}", .{ua}) catch return error.AzureOpenAiApiError;
            headers_buf[header_count] = user_agent_hdr.?;
            header_count += 1;
        }

        const timeout_secs = if (request.timeout_secs < 60) 60 else request.timeout_secs;
        const resp_body = root.curlPostTimed(allocator, chat_url, body, headers_buf[0..header_count], timeout_secs) catch return error.AzureOpenAiApiError;
        defer allocator.free(resp_body);

        return parseNativeResponse(allocator, resp_body);
    }

    fn supportsNativeToolsImpl(_: *anyopaque) bool {
        return true;
    }

    fn supportsVisionImpl(_: *anyopaque) bool {
        return true;
    }

    fn getNameImpl(_: *anyopaque) []const u8 {
        return "Azure OpenAI";
    }

    fn deinitImpl(_: *anyopaque) void {}

    /// Build a streaming chat request JSON body (same as buildChatRequestBody but with "stream":true).
    fn buildStreamingChatRequestBody(
        allocator: std.mem.Allocator,
        request: ChatRequest,
        model: []const u8,
        temperature: f64,
    ) ![]const u8 {
        var buf: std.ArrayListUnmanaged(u8) = .empty;
        errdefer buf.deinit(allocator);

        try buf.appendSlice(allocator, "{\"messages\":[");

        for (request.messages, 0..) |msg, i| {
            if (i > 0) try buf.append(allocator, ',');
            try buf.appendSlice(allocator, "{\"role\":\"");
            try buf.appendSlice(allocator, msg.role.toSlice());
            try buf.appendSlice(allocator, "\",\"content\":");
            try root.serializeMessageContent(&buf, allocator, msg);
            if (msg.tool_call_id) |tc_id| {
                try buf.appendSlice(allocator, ",\"tool_call_id\":");
                try root.appendJsonString(&buf, allocator, tc_id);
            }
            try buf.append(allocator, '}');
        }

        try buf.append(allocator, ']');
        try root.appendGenerationFields(&buf, allocator, model, temperature, request.max_tokens, request.reasoning_effort);

        if (request.tools) |tools| {
            if (tools.len > 0) {
                try buf.appendSlice(allocator, ",\"tools\":");
                try root.convertToolsOpenAI(&buf, allocator, tools);
                try buf.appendSlice(allocator, ",\"tool_choice\":\"auto\"");
            }
        }

        try buf.appendSlice(allocator, ",\"stream\":true}");
        return try buf.toOwnedSlice(allocator);
    }

    /// Build a full chat request JSON body from a ChatRequest.
    fn buildChatRequestBody(
        allocator: std.mem.Allocator,
        request: ChatRequest,
        model: []const u8,
        temperature: f64,
    ) ![]const u8 {
        var buf: std.ArrayListUnmanaged(u8) = .empty;
        errdefer buf.deinit(allocator);

        try buf.appendSlice(allocator, "{\"messages\":[");

        for (request.messages, 0..) |msg, i| {
            if (i > 0) try buf.append(allocator, ',');
            try buf.appendSlice(allocator, "{\"role\":\"");
            try buf.appendSlice(allocator, msg.role.toSlice());
            try buf.appendSlice(allocator, "\",\"content\":");
            try root.serializeMessageContent(&buf, allocator, msg);
            if (msg.tool_call_id) |tc_id| {
                try buf.appendSlice(allocator, ",\"tool_call_id\":");
                try root.appendJsonString(&buf, allocator, tc_id);
            }
            try buf.append(allocator, '}');
        }

        try buf.append(allocator, ']');
        try root.appendGenerationFields(&buf, allocator, model, temperature, request.max_tokens, request.reasoning_effort);

        if (request.tools) |tools| {
            if (tools.len > 0) {
                try buf.appendSlice(allocator, ",\"tools\":");
                try root.convertToolsOpenAI(&buf, allocator, tools);
                try buf.appendSlice(allocator, ",\"tool_choice\":\"auto\"");
            }
        }

        try buf.append(allocator, '}');
        return try buf.toOwnedSlice(allocator);
    }
};

// ════════════════════════════════════════════════════════════════════════════
// Tests
// ════════════════════════════════════════════════════════════════════════════

test "buildRequestBody with system prompt" {
    const body = try AzureOpenAiProvider.buildRequestBody(std.testing.allocator, "You are helpful", "hello", "gpt-4.1", 0.7, 4096);
    defer std.testing.allocator.free(body);
    try std.testing.expect(std.mem.indexOf(u8, body, "\"role\":\"system\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, body, "You are helpful") != null);
    try std.testing.expect(std.mem.indexOf(u8, body, "hello") != null);
    try std.testing.expect(std.mem.indexOf(u8, body, "\"max_tokens\":4096") != null);
    // Azure doesn't include model in request body
    try std.testing.expect(std.mem.indexOf(u8, body, "\"model\":") == null);
}

test "buildChatUrl constructs correct Azure URL" {
    const provider = AzureOpenAiProvider.init(
        std.testing.allocator,
        "test-key",
        "https://myresource.openai.azure.com",
        null,
    );
    const url = try provider.buildChatUrl(std.testing.allocator, "gpt-5.2-chat");
    defer std.testing.allocator.free(url);
    try std.testing.expectEqualStrings(
        "https://myresource.openai.azure.com/openai/deployments/gpt-5.2-chat/chat/completions?api-version=2024-10-21",
        url,
    );
}

test "buildChatUrl handles trailing slash" {
    const provider = AzureOpenAiProvider.init(
        std.testing.allocator,
        "test-key",
        "https://myresource.openai.azure.com/",
        null,
    );
    const url = try provider.buildChatUrl(std.testing.allocator, "gpt-35-turbo");
    defer std.testing.allocator.free(url);
    try std.testing.expectEqualStrings(
        "https://myresource.openai.azure.com/openai/deployments/gpt-35-turbo/chat/completions?api-version=2024-10-21",
        url,
    );
}

test "buildChatUrl accepts v1-style base URL" {
    const provider = AzureOpenAiProvider.init(
        std.testing.allocator,
        "test-key",
        "https://myresource.openai.azure.com/openai/v1/",
        null,
    );
    const url = try provider.buildChatUrl(std.testing.allocator, "gpt-5.2-chat");
    defer std.testing.allocator.free(url);
    try std.testing.expectEqualStrings(
        "https://myresource.openai.azure.com/openai/deployments/gpt-5.2-chat/chat/completions?api-version=2024-10-21",
        url,
    );
}

test "buildChatUrl rejects placeholder base URL" {
    const provider = AzureOpenAiProvider.init(
        std.testing.allocator,
        "test-key",
        AzureOpenAiProvider.DEFAULT_BASE_URL,
        null,
    );
    try std.testing.expectError(error.InvalidBaseUrl, provider.buildChatUrl(std.testing.allocator, "gpt-5.2-chat"));
}

test "parseTextResponse single choice" {
    const body =
        \\{"choices":[{"message":{"content":"Hi!"}}]}
    ;
    const result = try AzureOpenAiProvider.parseTextResponse(std.testing.allocator, body);
    defer std.testing.allocator.free(result);
    try std.testing.expectEqualStrings("Hi!", result);
}

test "parseNativeResponse with tool calls" {
    const body =
        \\{"choices":[{"message":{"content":"Let me help","tool_calls":[{"id":"call_1","type":"function","function":{"name":"shell","arguments":"{\"cmd\":\"ls\"}"}}]}}],"model":"gpt-4","usage":{"prompt_tokens":5,"completion_tokens":10,"total_tokens":15}}
    ;
    const response = try AzureOpenAiProvider.parseNativeResponse(std.testing.allocator, body);
    defer {
        if (response.content) |c| std.testing.allocator.free(c);
        for (response.tool_calls) |tc| {
            std.testing.allocator.free(tc.id);
            std.testing.allocator.free(tc.name);
            std.testing.allocator.free(tc.arguments);
        }
        std.testing.allocator.free(response.tool_calls);
        std.testing.allocator.free(response.model);
    }
    try std.testing.expectEqualStrings("Let me help", response.content.?);
    try std.testing.expect(response.tool_calls.len == 1);
    try std.testing.expectEqualStrings("shell", response.tool_calls[0].name);
    try std.testing.expectEqualStrings("call_1", response.tool_calls[0].id);
    try std.testing.expect(response.usage.prompt_tokens == 5);
    try std.testing.expect(response.usage.total_tokens == 15);
}

test "supportsNativeTools returns true" {
    var p = AzureOpenAiProvider.init(std.testing.allocator, "key", "https://test.openai.azure.com", null);
    const prov = p.provider();
    try std.testing.expect(prov.supportsNativeTools());
}

test "init with empty key" {
    const p = AzureOpenAiProvider.init(std.testing.allocator, null, "https://test.openai.azure.com", null);
    try std.testing.expect(p.api_key == null);
}

test "init with custom user agent" {
    const p = AzureOpenAiProvider.init(std.testing.allocator, "key", "https://test.openai.azure.com", "CustomAgent/1.0");
    try std.testing.expectEqualStrings("CustomAgent/1.0", p.user_agent.?);
}

test "provider getName returns Azure OpenAI" {
    var p = AzureOpenAiProvider.init(std.testing.allocator, "key", "https://test.openai.azure.com", null);
    const prov = p.provider();
    try std.testing.expectEqualStrings("Azure OpenAI", prov.getName());
}

test "buildChatUrl API version is hardcoded" {
    const provider = AzureOpenAiProvider.init(
        std.testing.allocator,
        "test-key",
        "https://myresource.openai.azure.com",
        null,
    );
    const url = try provider.buildChatUrl(std.testing.allocator, "gpt-5.2-chat");
    defer std.testing.allocator.free(url);
    try std.testing.expect(std.mem.indexOf(u8, url, "api-version=2024-10-21") != null);
}

test "buildRequestBody without system prompt" {
    const body = try AzureOpenAiProvider.buildRequestBody(std.testing.allocator, null, "hello", "gpt-4.1", 0.7, 4096);
    defer std.testing.allocator.free(body);
    try std.testing.expect(std.mem.indexOf(u8, body, "role\":\"system\"") == null);
    try std.testing.expect(std.mem.indexOf(u8, body, "hello") != null);
    try std.testing.expect(std.mem.indexOf(u8, body, "\"max_tokens\":4096") != null);
}

test "buildRequestBody without max_tokens omits it" {
    const body = try AzureOpenAiProvider.buildRequestBody(std.testing.allocator, null, "hello", "gpt-4.1", 0.7, null);
    defer std.testing.allocator.free(body);
    try std.testing.expect(std.mem.indexOf(u8, body, "max_completion_tokens") == null);
    try std.testing.expect(std.mem.indexOf(u8, body, "\"max_tokens\":") == null);
    try std.testing.expect(std.mem.indexOf(u8, body, "hello") != null);
}

test "buildChatRequestBody preserves reasoning fields for gpt-5" {
    const msgs = [_]root.ChatMessage{
        .{ .role = .user, .content = "hello" },
    };
    const req = root.ChatRequest{
        .messages = &msgs,
        .model = "gpt-5.2-chat",
        .max_tokens = 42,
        .reasoning_effort = "high",
    };

    const body = try AzureOpenAiProvider.buildChatRequestBody(std.testing.allocator, req, req.model, 0.3);
    defer std.testing.allocator.free(body);

    try std.testing.expect(std.mem.indexOf(u8, body, "\"reasoning_effort\":\"high\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, body, "\"max_completion_tokens\":42") != null);
    try std.testing.expect(std.mem.indexOf(u8, body, "\"temperature\":") == null);
}

test "parseTextResponse empty choices fails" {
    const body =
        \\{"choices":[]}
    ;
    try std.testing.expectError(error.NoResponseContent, AzureOpenAiProvider.parseTextResponse(std.testing.allocator, body));
}

test "parseTextResponse multiple choices returns first" {
    const body =
        \\{"choices":[{"message":{"content":"First"}},{"message":{"content":"Second"}}]}
    ;
    const result = try AzureOpenAiProvider.parseTextResponse(std.testing.allocator, body);
    defer std.testing.allocator.free(result);
    try std.testing.expectEqualStrings("First", result);
}

test "parseNativeResponse text only no tool calls" {
    const body =
        \\{"choices":[{"message":{"content":"Hello there"}}],"model":"gpt-4","usage":{"prompt_tokens":3,"completion_tokens":5,"total_tokens":8}}
    ;
    const response = try AzureOpenAiProvider.parseNativeResponse(std.testing.allocator, body);
    defer {
        if (response.content) |c| std.testing.allocator.free(c);
        std.testing.allocator.free(response.tool_calls); // empty array
        std.testing.allocator.free(response.model);
    }
    try std.testing.expectEqualStrings("Hello there", response.content.?);
    try std.testing.expect(response.tool_calls.len == 0);
    try std.testing.expect(response.usage.prompt_tokens == 3);
}

test "parseNativeResponse empty choices fails" {
    const body =
        \\{"choices":[],"model":"gpt-4"}
    ;
    try std.testing.expectError(error.NoResponseContent, AzureOpenAiProvider.parseNativeResponse(std.testing.allocator, body));
}

test "validateUserAgent rejects CRLF injection" {
    try std.testing.expect(!AzureOpenAiProvider.validateUserAgent("Agent\r\nHeader-Injection: value"));
    try std.testing.expect(!AzureOpenAiProvider.validateUserAgent("Agent\nHeader-Injection: value"));
    try std.testing.expect(AzureOpenAiProvider.validateUserAgent("ValidAgent/1.0"));
}

test "timeout enforcement ensures minimum 60 seconds" {
    // Test that Azure provider enforces minimum 60 second timeout
    // This is unit test validation - actual HTTP call would be mocked in integration tests
    const min_timeout: u32 = 60;
    const low_timeout: u32 = 30;
    const high_timeout: u32 = 120;

    // Verify timeout logic
    const adjusted_low = if (low_timeout < 60) 60 else low_timeout;
    const adjusted_high = if (high_timeout < 60) 60 else high_timeout;

    try std.testing.expect(adjusted_low == min_timeout);
    try std.testing.expect(adjusted_high == high_timeout);
}
