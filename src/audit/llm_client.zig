//! Envelope-triage call over the project's existing provider abstraction.
//!
//! This module does not construct providers. Callers pass the same provider
//! vtable interface the agent uses, and this layer only owns the triage prompt
//! plus verdict parsing.

const std = @import("std");
const Allocator = std.mem.Allocator;
const providers = @import("../providers/root.zig");
const util = @import("../util.zig");

const MAX_REASONING_BYTES: usize = 200;

pub const Decision = enum {
    real_secret,
    false_positive,
    uncertain,

    pub fn parse(text: []const u8) Decision {
        if (std.mem.eql(u8, text, "real_secret")) return .real_secret;
        if (std.mem.eql(u8, text, "false_positive")) return .false_positive;
        return .uncertain;
    }

    pub fn name(self: Decision) []const u8 {
        return switch (self) {
            .real_secret => "real_secret",
            .false_positive => "false_positive",
            .uncertain => "uncertain",
        };
    }
};

pub const Verdict = struct {
    decision: Decision,
    severity_adjusted: []u8,
    reasoning: []u8,
    confidence_score: f64,

    pub fn deinit(self: *Verdict, allocator: Allocator) void {
        allocator.free(self.severity_adjusted);
        allocator.free(self.reasoning);
    }
};

pub const TriageClient = struct {
    ptr: *anyopaque,
    vtable: *const VTable,

    pub const VTable = struct {
        triageEnvelope: *const fn (*anyopaque, Allocator, []const u8) anyerror!Verdict,
    };

    pub fn triageEnvelope(self: TriageClient, allocator: Allocator, envelope_json: []const u8) !Verdict {
        return self.vtable.triageEnvelope(self.ptr, allocator, envelope_json);
    }
};

pub const ProviderTriageClient = struct {
    provider: providers.Provider,
    model: []const u8,
    temperature: f64,

    pub fn client(self: *ProviderTriageClient) TriageClient {
        return .{
            .ptr = self,
            .vtable = &vtable,
        };
    }

    fn triageEnvelopeImpl(ptr: *anyopaque, allocator: Allocator, envelope_json: []const u8) !Verdict {
        const self: *ProviderTriageClient = @ptrCast(@alignCast(ptr));
        return triageEnvelope(allocator, self.provider, self.model, self.temperature, envelope_json);
    }

    const vtable = TriageClient.VTable{
        .triageEnvelope = triageEnvelopeImpl,
    };
};

const SYSTEM_PROMPT =
    \\You are a security analyst triaging potential secret leaks.
    \\You receive a privacy-preserving JSON envelope describing a candidate
    \\secret found in a code repository. The envelope NEVER contains the raw
    \\secret value — only its shape (length, charset, entropy), its location,
    \\and surrounding context.
    \\
    \\Your job: decide whether this is a real secret leak, a false positive,
    \\or uncertain. Respond with ONLY a JSON object on a single line, no
    \\preamble, no markdown, no code fences.
    \\
    \\Schema:
    \\{"decision":"real_secret"|"false_positive"|"uncertain","severity_adjusted":"critical"|"high"|"medium"|"low","reasoning":"brief explanation under 200 chars","confidence_score":0.0-1.0}
    \\
    \\Rules of thumb:
    \\- token_type_fingerprint is a strong signal: github_pat / aws_access_key_id / slack_bot_token / openai_project / stripe_live etc. → almost always real_secret unless is_test_path or is_example_file is true.
    \\- pem_private_key → critical real_secret unless is_example_file.
    \\- is_test_path=true or is_example_file=true → likely false_positive even with real-looking shape.
    \\- nearby_keywords containing "example", "fake", "placeholder", "sample" → bias toward false_positive.
    \\- Short length (<16) + low entropy (<3.0) + no fingerprint → false_positive (it's a placeholder).
    \\- High entropy (>4.0) + length >=20 + non-test path + no example markers → real_secret.
    \\- is_in_comment=true or is_in_docstring=true → biased toward false_positive (it's an example).
;

/// Submit an envelope to the configured provider and return a parsed verdict.
/// Caller owns the returned Verdict and must call deinit on it.
pub fn triageEnvelope(
    allocator: Allocator,
    provider: providers.Provider,
    model: []const u8,
    temperature: f64,
    envelope_json: []const u8,
) !Verdict {
    const response_text = try provider.chatWithSystem(
        allocator,
        SYSTEM_PROMPT,
        envelope_json,
        model,
        temperature,
    );
    defer allocator.free(response_text);

    const trimmed = std.mem.trim(u8, response_text, " \t\r\n");
    return parseVerdictJson(allocator, stripFences(trimmed));
}

fn stripFences(text: []const u8) []const u8 {
    var s = text;
    if (std.mem.startsWith(u8, s, "```json")) {
        s = s[7..];
    } else if (std.mem.startsWith(u8, s, "```")) {
        s = s[3..];
    }
    if (std.mem.endsWith(u8, s, "```")) {
        s = s[0 .. s.len - 3];
    }
    return std.mem.trim(u8, s, " \t\r\n");
}

pub fn parseVerdictJson(allocator: Allocator, text: []const u8) !Verdict {
    var parsed = try std.json.parseFromSlice(std.json.Value, allocator, text, .{});
    defer parsed.deinit();

    const root = parsed.value;
    if (root != .object) return error.InvalidVerdict;

    const decision_str = if (root.object.get("decision")) |d|
        (if (d == .string) d.string else "uncertain")
    else
        "uncertain";
    const decision = Decision.parse(decision_str);

    const raw_severity_str = if (root.object.get("severity_adjusted")) |s|
        (if (s == .string) s.string else "medium")
    else
        "medium";
    const severity_str = normalizeSeverity(raw_severity_str);
    const severity_dup = try allocator.dupe(u8, severity_str);
    errdefer allocator.free(severity_dup);

    const reasoning_str = if (root.object.get("reasoning")) |r|
        (if (r == .string) r.string else "")
    else
        "";
    const reasoning_preview = util.previewUtf8(reasoning_str, MAX_REASONING_BYTES);
    const reasoning_dup = try allocator.dupe(u8, reasoning_preview.slice);
    errdefer allocator.free(reasoning_dup);

    const raw_confidence: f64 = if (root.object.get("confidence_score")) |c| switch (c) {
        .float => c.float,
        .integer => @floatFromInt(c.integer),
        else => 0.5,
    } else 0.5;
    const confidence = normalizeConfidence(raw_confidence);

    return .{
        .decision = decision,
        .severity_adjusted = severity_dup,
        .reasoning = reasoning_dup,
        .confidence_score = confidence,
    };
}

fn normalizeSeverity(text: []const u8) []const u8 {
    if (std.mem.eql(u8, text, "critical")) return text;
    if (std.mem.eql(u8, text, "high")) return text;
    if (std.mem.eql(u8, text, "medium")) return text;
    if (std.mem.eql(u8, text, "low")) return text;
    return "medium";
}

fn normalizeConfidence(value: f64) f64 {
    if (!std.math.isFinite(value)) return 0.5;
    if (value < 0.0) return 0.0;
    if (value > 1.0) return 1.0;
    return value;
}

test "parseVerdictJson valid" {
    const allocator = std.testing.allocator;
    const text =
        \\{"decision":"real_secret","severity_adjusted":"high","reasoning":"github_pat in non-test config","confidence_score":0.95}
    ;
    var v = try parseVerdictJson(allocator, text);
    defer v.deinit(allocator);
    try std.testing.expectEqual(Decision.real_secret, v.decision);
    try std.testing.expectEqualStrings("high", v.severity_adjusted);
    try std.testing.expect(v.confidence_score > 0.9);
}

test "parseVerdictJson false_positive" {
    const allocator = std.testing.allocator;
    const text =
        \\{"decision":"false_positive","severity_adjusted":"low","reasoning":"placeholder in test fixture","confidence_score":0.9}
    ;
    var v = try parseVerdictJson(allocator, text);
    defer v.deinit(allocator);
    try std.testing.expectEqual(Decision.false_positive, v.decision);
}

test "parseVerdictJson normalizes model-controlled fields" {
    const allocator = std.testing.allocator;
    const long_reason = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
    const text = try std.fmt.allocPrint(
        allocator,
        "{{\"decision\":\"real_secret\",\"severity_adjusted\":\"bogus\",\"reasoning\":\"{s}\",\"confidence_score\":42.0}}",
        .{long_reason},
    );
    defer allocator.free(text);

    var v = try parseVerdictJson(allocator, text);
    defer v.deinit(allocator);

    try std.testing.expectEqual(Decision.real_secret, v.decision);
    try std.testing.expectEqualStrings("medium", v.severity_adjusted);
    try std.testing.expect(v.reasoning.len <= MAX_REASONING_BYTES);
    try std.testing.expectEqual(@as(f64, 1.0), v.confidence_score);
}

test "provider triage client adapts provider vtable" {
    const testing_allocator = std.testing.allocator;

    const TestProvider = struct {
        called: bool = false,

        fn chatWithSystem(
            ptr: *anyopaque,
            allocator: Allocator,
            _: ?[]const u8,
            message: []const u8,
            model: []const u8,
            temperature: f64,
        ) ![]const u8 {
            const self: *@This() = @ptrCast(@alignCast(ptr));
            self.called = true;
            try std.testing.expectEqualStrings("{\"x\":1}", message);
            try std.testing.expectEqualStrings("triage-model", model);
            try std.testing.expectEqual(@as(f64, 0.0), temperature);
            return allocator.dupe(u8, "{\"decision\":\"uncertain\",\"severity_adjusted\":\"medium\",\"reasoning\":\"ok\",\"confidence_score\":0.5}");
        }

        fn chat(_: *anyopaque, _: Allocator, _: providers.ChatRequest, _: []const u8, _: f64) !providers.ChatResponse {
            return error.NotSupported;
        }

        fn supportsNativeTools(_: *anyopaque) bool {
            return false;
        }

        fn getName(_: *anyopaque) []const u8 {
            return "test";
        }

        fn deinit(_: *anyopaque) void {}

        const vtable = providers.Provider.VTable{
            .chatWithSystem = chatWithSystem,
            .chat = chat,
            .supportsNativeTools = supportsNativeTools,
            .getName = getName,
            .deinit = deinit,
        };
    };

    var test_provider = TestProvider{};
    const provider = providers.Provider{
        .ptr = &test_provider,
        .vtable = &TestProvider.vtable,
    };
    var provider_client = ProviderTriageClient{
        .provider = provider,
        .model = "triage-model",
        .temperature = 0.0,
    };

    var verdict = try provider_client.client().triageEnvelope(testing_allocator, "{\"x\":1}");
    defer verdict.deinit(testing_allocator);

    try std.testing.expect(test_provider.called);
    try std.testing.expectEqual(Decision.uncertain, verdict.decision);
}

test "stripFences removes markdown code fence" {
    const text = "```json\n{\"decision\":\"uncertain\"}\n```";
    const stripped = stripFences(text);
    try std.testing.expect(std.mem.startsWith(u8, stripped, "{"));
    try std.testing.expect(std.mem.endsWith(u8, stripped, "}"));
}

test "triageEnvelope uses supplied provider interface" {
    const TestProvider = struct {
        called: bool = false,

        fn chatWithSystem(
            ptr: *anyopaque,
            allocator: Allocator,
            system_prompt: ?[]const u8,
            message: []const u8,
            model: []const u8,
            temperature: f64,
        ) ![]const u8 {
            const self: *@This() = @ptrCast(@alignCast(ptr));
            self.called = true;
            try std.testing.expect(system_prompt != null);
            try std.testing.expectEqualStrings("{}", message);
            try std.testing.expectEqualStrings("triage-model", model);
            try std.testing.expectEqual(@as(f64, 0.0), temperature);
            return allocator.dupe(u8, "{\"decision\":\"false_positive\",\"severity_adjusted\":\"low\",\"reasoning\":\"test\",\"confidence_score\":0.9}");
        }

        fn chat(_: *anyopaque, _: Allocator, _: providers.ChatRequest, _: []const u8, _: f64) !providers.ChatResponse {
            return error.NotSupported;
        }

        fn supportsNativeTools(_: *anyopaque) bool {
            return false;
        }

        fn getName(_: *anyopaque) []const u8 {
            return "test";
        }

        fn deinit(_: *anyopaque) void {}

        const vtable = providers.Provider.VTable{
            .chatWithSystem = chatWithSystem,
            .chat = chat,
            .supportsNativeTools = supportsNativeTools,
            .getName = getName,
            .deinit = deinit,
        };
    };

    var test_provider = TestProvider{};
    const provider = providers.Provider{
        .ptr = &test_provider,
        .vtable = &TestProvider.vtable,
    };

    var verdict = try triageEnvelope(std.testing.allocator, provider, "triage-model", 0.0, "{}");
    defer verdict.deinit(std.testing.allocator);

    try std.testing.expect(test_provider.called);
    try std.testing.expectEqual(Decision.false_positive, verdict.decision);
    try std.testing.expectEqualStrings("low", verdict.severity_adjusted);
}
