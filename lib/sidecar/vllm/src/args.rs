// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use dynamo_sidecar_common::{HttpEndpoint, SidecarArgs};

use std::num::NonZeroU32;

fn parse_http_endpoint(raw: &str) -> Result<HttpEndpoint, String> {
    HttpEndpoint::parse(raw, "--vllm-http-endpoint").map_err(|error| error.to_string())
}

/// The vLLM KV-transfer connector the engine runs (`--kv-transfer-config`'s
/// `kv_connector`). Connectors disagree on who builds the PD handoff:
/// NixlConnector's prefill terminal response carries the decode-side
/// parameters, while MooncakeConnector's never does, so the sidecar allocates
/// a per-request transfer_id and publishes the handoff itself.
#[derive(clap::ValueEnum, Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) enum KvConnector {
    #[default]
    #[value(name = "NixlConnector")]
    Nixl,
    #[value(name = "MooncakeConnector")]
    Mooncake,
}

/// vLLM's default VLLM_MOONCAKE_BOOTSTRAP_PORT.
pub(crate) const DEFAULT_MOONCAKE_BOOTSTRAP_PORT: u16 = 8998;

#[derive(clap::Parser, Clone, Debug)]
#[command(
    name = "dynamo-vllm-sidecar",
    about = "Run a Dynamo worker against vLLM's native gRPC service"
)]
pub(crate) struct Args {
    #[command(flatten)]
    pub sidecar: SidecarArgs,

    /// Optional controller-routable vLLM HTTP base URL for RL compatibility operations.
    #[arg(
        long,
        env = "VLLM_HTTP_ENDPOINT",
        value_parser = parse_http_endpoint
    )]
    pub vllm_http_endpoint: Option<HttpEndpoint>,

    /// Optional total RL process world size for vLLM versions that omit it from gRPC metadata.
    ///
    /// The value includes tensor, pipeline, prefill-context, and data parallelism.
    #[arg(long, env = "DYN_VLLM_RL_WORLD_SIZE")]
    pub vllm_rl_world_size: Option<NonZeroU32>,

    /// The engine's KV-transfer connector. Only prefill workers behave
    /// differently: MooncakeConnector needs the sidecar to drive the PD
    /// handoff because the engine never returns kv_transfer_params.
    #[arg(
        long,
        env = "DYN_VLLM_KV_CONNECTOR",
        value_enum,
        default_value = "NixlConnector"
    )]
    pub kv_connector: KvConnector,

    /// Host the decode worker uses to reach this pod's Mooncake bootstrap
    /// server, typically injected from `status.podIP`. Required when a prefill
    /// worker runs `--kv-connector MooncakeConnector`.
    #[arg(long, env = "DYN_VLLM_MOONCAKE_BOOTSTRAP_HOST")]
    pub mooncake_bootstrap_host: Option<String>,

    /// Mooncake bootstrap server port. Shares the engine's
    /// VLLM_MOONCAKE_BOOTSTRAP_PORT variable so the two stay in sync.
    #[arg(
        long,
        env = "VLLM_MOONCAKE_BOOTSTRAP_PORT",
        default_value_t = DEFAULT_MOONCAKE_BOOTSTRAP_PORT
    )]
    pub mooncake_bootstrap_port: u16,
}
