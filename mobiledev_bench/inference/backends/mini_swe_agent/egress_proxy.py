"""Egress-filtering proxy: lets agent containers reach package registries (for build/dependency
resolution) while blocking everything else - in particular GitHub and search engines, the two
realistic sources of gold-fix/solution leakage now that a blanket `--network none` has been
dropped (see runner.py's run_args comment for why).

This is a genuine network-topology guarantee, not a prompt-level one: the agent's container is
attached ONLY to a `docker network create --internal` network, which Docker gives no route to the
internet at all. The only way off that network is through a Squid proxy container that straddles
both the internal network and the default bridge (so it alone has real internet access), with
Squid's own ACLs allowlisting a fixed set of registry domains and denying everything else by
default. A tool that ignores the http_proxy/https_proxy env vars entirely doesn't get through some
looser fallback - it simply has no path to the internet to fall back to.

Both the network and the proxy container are created once, lazily, and left running - shared
across every instance in a run_inference.py invocation (and reused by future invocations) rather
than paying Squid startup cost per instance. Tear down manually if you want to force it to pick up
an allowlist change:
    docker rm -f mobiledev-bench-mini-swe-agent-egress-proxy
    docker network rm mobiledev-bench-mini-swe-agent-egress
"""

import logging
import subprocess
import tempfile
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

NETWORK_NAME = "mobiledev-bench-mini-swe-agent-egress"
PROXY_CONTAINER_NAME = "mobiledev-bench-mini-swe-agent-egress-proxy"
PROXY_IMAGE = "ubuntu/squid:latest"
PROXY_PORT = 3128

DEFAULT_ALLOWED_DOMAINS = [
    # Gradle/Maven (Kotlin, Java)
    "dl.google.com",
    "repo.maven.apache.org",
    "repo1.maven.org",
    "plugins.gradle.org",
    "services.gradle.org",
    "repo.gradle.org",
    "jcenter.bintray.com",
    "jitpack.io",
    # npm (TypeScript / React Native)
    "registry.npmjs.org",
    "registry.yarnpkg.com",
    # pub.dev (Dart / Flutter)
    "pub.dev",
    "pub.dartlang.org",
    "storage.googleapis.com",
]

_lock = threading.Lock()


@dataclass
class EgressProxyInfo:
    network_name: str
    proxy_url: str
    env: dict = field(default_factory=dict)
    """http_proxy/https_proxy-family env vars to set for every command run inside the agent's
    container - covers tools that read the env var convention (curl, npm, pub, pip, most others)."""


def _squid_conf(allowed_domains: list[str]) -> str:
    domain_acl = " ".join(f".{d}" for d in allowed_domains)
    return f"""\
acl allowed_domains dstdomain {domain_acl}
acl SSL_ports port 443
acl Safe_ports port 80 443
acl CONNECT method CONNECT

http_access deny !Safe_ports
http_access deny CONNECT !SSL_ports
http_access allow allowed_domains
http_access deny all

http_port {PROXY_PORT}
coredump_dir /var/spool/squid
"""


def _run(cmd: list[str], **kwargs) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, **kwargs)


def _network_exists(name: str) -> bool:
    return _run(["docker", "network", "inspect", name]).returncode == 0


def _container_running(name: str) -> bool:
    result = _run(["docker", "inspect", "-f", "{{.State.Running}}", name])
    return result.returncode == 0 and result.stdout.strip() == "true"


def ensure_egress_proxy(
    allowed_domains: Optional[list[str]] = None,
    logger: Optional[logging.Logger] = None,
) -> EgressProxyInfo:
    """Idempotently ensure the internal network + Squid proxy exist and are running, then return
    connection info for attaching an agent's container to them. Safe to call concurrently from
    multiple ThreadPoolExecutor workers - serialized by a module-level lock so only the first
    caller actually creates anything."""
    log = logger or logging.getLogger("mini_swe_agent.egress_proxy")
    domains = allowed_domains if allowed_domains is not None else DEFAULT_ALLOWED_DOMAINS

    with _lock:
        if not _network_exists(NETWORK_NAME):
            log.info(f"Creating internal egress network '{NETWORK_NAME}'")
            result = _run(["docker", "network", "create", "--internal", NETWORK_NAME])
            if result.returncode != 0:
                raise RuntimeError(f"Failed to create egress network: {result.stderr}")

        if not _container_running(PROXY_CONTAINER_NAME):
            # Remove any stopped/stale container left over from a prior crashed run before
            # recreating - `docker run --name` fails if a stopped container already owns the name.
            _run(["docker", "rm", "-f", PROXY_CONTAINER_NAME])

            conf_dir = Path(tempfile.gettempdir()) / "mobiledev-bench-egress-proxy"
            conf_dir.mkdir(parents=True, exist_ok=True)
            conf_path = conf_dir / "squid.conf"
            conf_path.write_text(_squid_conf(domains), encoding="utf-8")

            log.info(
                f"Starting egress proxy '{PROXY_CONTAINER_NAME}' "
                f"(allowlisting {len(domains)} domains)"
            )
            result = _run(
                [
                    "docker",
                    "run",
                    "-d",
                    "--name",
                    PROXY_CONTAINER_NAME,
                    "--network",
                    NETWORK_NAME,
                    "-v",
                    f"{conf_path}:/etc/squid/squid.conf:ro",
                    PROXY_IMAGE,
                ]
            )
            if result.returncode != 0:
                raise RuntimeError(f"Failed to start egress proxy: {result.stderr}")

            # The proxy needs a route to the real internet too, in addition to the internal
            # network the agent's container is confined to - connect it to the default bridge as
            # a second interface. The agent's container is never attached to this bridge itself.
            result = _run(["docker", "network", "connect", "bridge", PROXY_CONTAINER_NAME])
            if result.returncode != 0:
                log.warning(f"Failed to connect egress proxy to bridge network: {result.stderr}")

        proxy_url = f"http://{PROXY_CONTAINER_NAME}:{PROXY_PORT}"
        return EgressProxyInfo(
            network_name=NETWORK_NAME,
            proxy_url=proxy_url,
            env={
                "http_proxy": proxy_url,
                "https_proxy": proxy_url,
                "HTTP_PROXY": proxy_url,
                "HTTPS_PROXY": proxy_url,
                "no_proxy": "localhost,127.0.0.1",
                "NO_PROXY": "localhost,127.0.0.1",
            },
        )
