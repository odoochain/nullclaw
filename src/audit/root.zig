//! Audit subsystem — privacy-preserving secret triage.
//!
//! Sub-modules:
//!   envelope.zig    — privacy-safe metadata envelopes for findings
//!   llm_client.zig  — provider-vtable envelope triage prompt/parser
//!   audit_log.zig   — append-only log of LLM-triage requests
//!   triager.zig     — orchestration: envelope → LLM → log → verdict

pub const types = @import("types.zig");
pub const envelope = @import("envelope.zig");
pub const llm_client = @import("llm_client.zig");
pub const audit_log = @import("audit_log.zig");
pub const triager = @import("triager.zig");

pub const Envelope = envelope.Envelope;
pub const BuildInput = envelope.BuildInput;
pub const Charset = envelope.Charset;
pub const TokenTypeFingerprint = envelope.TokenTypeFingerprint;
pub const Severity = types.Severity;
pub const Confidence = types.Confidence;
pub const FailureThreshold = types.FailureThreshold;
pub const FindingSource = types.FindingSource;
pub const TriageMode = types.TriageMode;
pub const Finding = types.Finding;
pub const Report = types.Report;
pub const Verdict = llm_client.Verdict;
pub const Decision = llm_client.Decision;
pub const AuditLog = audit_log.AuditLog;
pub const TriageStats = types.TriageStats;

test {
    _ = types;
    _ = envelope;
    _ = llm_client;
    _ = audit_log;
    _ = triager;
}
