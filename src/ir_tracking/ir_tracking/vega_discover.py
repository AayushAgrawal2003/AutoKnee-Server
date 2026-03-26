"""
Polaris Vega discovery and connection module.

Scans for an NDI Polaris Vega camera on the network using multiple strategies:
  1. Bonjour/mDNS service browsing (macOS dns-sd)
  2. Hostname resolution (P9-XXXXX.local patterns from ARP/mDNS cache)
  3. Direct link-local subnet probe on port 8765
  4. Fallback to last-known / caller-supplied IP

Usage:
    from vega_discover import discover_vega, connect_vega

    # Just find the camera (returns connection dict)
    info = discover_vega()

    # Find + connect + start tracking (returns NDITracker)
    tracker = connect_vega(romfiles=["/path/to/tool.rom"])
"""

import socket
import subprocess
import re
import time
import concurrent.futures
from dataclasses import dataclass, field
from typing import Optional

import ndicapy

VEGA_DEFAULT_PORT = 8765
LINK_LOCAL_PREFIX = "169.254."
SOCKET_TIMEOUT = 0.4          # per-host TCP probe timeout (seconds)
BONJOUR_TIMEOUT = 3.0         # mDNS browse duration
SCAN_WORKERS = 64             # parallel threads for subnet scan
NDI_INIT_TIMEOUT = 2.0        # timeout for INIT: handshake


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class VegaConnectionInfo:
    ip: str
    port: int = VEGA_DEFAULT_PORT
    hostname: Optional[str] = None
    method: str = ""           # which discovery strategy succeeded
    firmware: str = ""

    def as_settings(self, romfiles: list[str]) -> dict:
        """Return a dict ready for sksurgerynditracker.NDITracker."""
        return {
            "tracker type": "vega",
            "ip address": self.ip,
            "port": self.port,
            "romfiles": romfiles,
        }

    def __str__(self):
        host = f" ({self.hostname})" if self.hostname else ""
        fw = f"  firmware={self.firmware}" if self.firmware else ""
        return f"Vega @ {self.ip}:{self.port}{host}  [found via {self.method}]{fw}"


class VegaNotFoundError(Exception):
    """Raised when no Polaris Vega could be found on any interface."""

    def __init__(self, attempts: list[str]):
        self.attempts = attempts
        summary = "\n  ".join(attempts) if attempts else "no discovery methods ran"
        super().__init__(
            f"Could not find a Polaris Vega camera.\nAttempted:\n  {summary}\n\n"
            "Troubleshooting:\n"
            "  - Is the Vega powered on and the front status LED green?\n"
            "  - Is an Ethernet cable connected directly (or via switch) to this Mac?\n"
            "  - Do you have a 169.254.x.x link-local address? (check: ifconfig)\n"
            "  - Try: ping 169.254.x.x  (the IP shown on the Vega LCD)\n"
        )


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------

def _tcp_probe(ip: str, port: int = VEGA_DEFAULT_PORT,
               timeout: float = SOCKET_TIMEOUT) -> bool:
    """Return True if a TCP connection to ip:port succeeds."""
    try:
        with socket.create_connection((ip, port), timeout=timeout):
            return True
    except (OSError, socket.timeout):
        return False


def _ndi_handshake(ip: str, port: int = VEGA_DEFAULT_PORT) -> Optional[str]:
    """Open an NDI connection, send INIT:, return firmware string or None."""
    device = None
    try:
        device = ndicapy.ndiOpenNetwork(ip, port)
        if not device:
            return None
        ndicapy.ndiCommand(device, "INIT:")
        err = ndicapy.ndiGetError(device)
        if err != ndicapy.NDI_OKAY:
            return None
        ndicapy.ndiCommand(device, "VER:0")
        err = ndicapy.ndiGetError(device)
        if err == ndicapy.NDI_OKAY:
            # VER:0 returns the full revision string
            ndicapy.ndiCommand(device, "VER:0")
            # Just mark as verified — firmware extraction is best-effort
            return "verified"
        return "verified"
    except Exception:
        return None
    finally:
        if device:
            try:
                ndicapy.ndiCloseNetwork(device)
            except Exception:
                pass


def _get_local_link_local_ips() -> list[str]:
    """Return all 169.254.x.x addresses assigned to local interfaces."""
    try:
        out = subprocess.run(["ifconfig"], capture_output=True, text=True,
                             timeout=5).stdout
    except Exception:
        return []
    return re.findall(r"inet (169\.254\.\d+\.\d+)", out)


def _arp_table_candidates() -> list[str]:
    """Pull resolved 169.254.x.x entries from the ARP cache (skip incomplete)."""
    try:
        out = subprocess.run(["arp", "-an"], capture_output=True, text=True,
                             timeout=5).stdout
    except Exception:
        return []
    # Only return IPs with a real MAC address (skip "(incomplete)" entries)
    candidates = []
    for line in out.splitlines():
        if "incomplete" in line:
            continue
        match = re.search(r"\(?(169\.254\.\d+\.\d+)\)?", line)
        if match:
            candidates.append(match.group(1))
    return candidates


# ---------------------------------------------------------------------------
# Discovery strategies
# ---------------------------------------------------------------------------

def _discover_bonjour(log: list[str]) -> Optional[VegaConnectionInfo]:
    """Use macOS dns-sd to browse for NDI Polaris Vega Bonjour services."""
    # The Vega advertises _pv-ndi._tcp and/or can be found via general browse.
    # We run dns-sd with a short timeout and parse any results.
    service_types = ["_pv-ndi._tcp", "_polaris._tcp", "_ndi._tcp"]

    for stype in service_types:
        try:
            proc = subprocess.Popen(
                ["dns-sd", "-B", stype, "local."],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            )
            time.sleep(BONJOUR_TIMEOUT)
            proc.terminate()
            stdout = proc.stdout.read()
        except Exception as e:
            log.append(f"Bonjour browse {stype}: failed ({e})")
            continue

        # Parse instance names  (e.g.  "P9-01075")
        # dns-sd -B output lines look like:
        #   Timestamp  A/R  Flags  if  Domain  Service  Instance
        instances = []
        for line in stdout.strip().split("\n"):
            parts = line.split()
            if len(parts) >= 7 and parts[1] in ("Add", "A"):
                instances.append(parts[-1])

        for instance in instances:
            hostname = f"{instance}.local"
            try:
                ip = socket.gethostbyname(hostname)
            except socket.gaierror:
                continue
            if _tcp_probe(ip):
                fw = _ndi_handshake(ip) or ""
                log.append(f"Bonjour browse {stype}: found {instance} -> {ip}")
                return VegaConnectionInfo(
                    ip=ip, hostname=hostname, method=f"bonjour ({stype})",
                    firmware=fw,
                )

        log.append(f"Bonjour browse {stype}: no instances found")

    return None


def _discover_hostname_resolve(log: list[str]) -> Optional[VegaConnectionInfo]:
    """Try resolving Vega hostnames from mDNS cache / ARP."""
    # Vega hostnames follow the pattern P9-XXXXX
    # Check ARP table for .local names and try resolving known patterns.
    try:
        out = subprocess.run(
            ["dns-sd", "-Q", "P9-01075.local", "A"],
            capture_output=True, text=True, timeout=4,
        )
        # Also try a general mDNS query for anything with P9- prefix
    except Exception:
        pass

    # Parse ARP for link-local entries and try them
    arp_ips = _arp_table_candidates()
    for ip in arp_ips:
        if _tcp_probe(ip):
            fw = _ndi_handshake(ip) or ""
            log.append(f"ARP cache: found Vega at {ip}")
            # Try reverse lookup for hostname
            hostname = None
            try:
                hostname = socket.gethostbyaddr(ip)[0]
            except Exception:
                pass
            return VegaConnectionInfo(
                ip=ip, hostname=hostname, method="arp_cache", firmware=fw,
            )

    log.append(f"ARP cache: checked {len(arp_ips)} link-local entries, none responded on :{VEGA_DEFAULT_PORT}")
    return None


def _discover_subnet_scan(log: list[str]) -> Optional[VegaConnectionInfo]:
    """Scan the local 169.254.x.x/16 subnet for port 8765.

    This is a brute-force fallback. To keep it fast we only scan a
    reasonable slice around the host's own link-local IP.
    """
    local_ips = _get_local_link_local_ips()
    if not local_ips:
        log.append("Subnet scan: no link-local address on this machine")
        return None

    # For each local link-local IP, scan the same /16 but prioritize
    # nearby addresses (same /24 first, then fan out)
    candidates = []
    seen = set()
    for local_ip in local_ips:
        parts = local_ip.split(".")
        base_b = int(parts[2])

        # Same /24 first (255 hosts), then adjacent /24s
        for offset in [0, 1, -1, 2, -2, 3, -3, 4, -4, 5, -5,
                       *range(6, 128), *range(-6, -128, -1)]:
            third = base_b + offset
            if third < 0 or third > 255:
                continue
            for fourth in range(1, 255):
                ip = f"169.254.{third}.{fourth}"
                if ip not in seen and ip not in local_ips:
                    seen.add(ip)
                    candidates.append(ip)

    total = len(candidates)
    log.append(f"Subnet scan: probing {total} link-local addresses on :{VEGA_DEFAULT_PORT} ...")

    found_ip = None

    def probe(ip):
        return ip if _tcp_probe(ip, timeout=SOCKET_TIMEOUT) else None

    with concurrent.futures.ThreadPoolExecutor(max_workers=SCAN_WORKERS) as pool:
        # Submit in batches so we can exit early
        batch_size = 512
        for start in range(0, total, batch_size):
            if found_ip:
                break
            batch = candidates[start:start + batch_size]
            futures = {pool.submit(probe, ip): ip for ip in batch}
            for fut in concurrent.futures.as_completed(futures):
                result = fut.result()
                if result:
                    found_ip = result
                    # Cancel remaining
                    for f in futures:
                        f.cancel()
                    break

    if found_ip:
        fw = _ndi_handshake(found_ip) or ""
        hostname = None
        try:
            hostname = socket.gethostbyaddr(found_ip)[0]
        except Exception:
            pass
        log.append(f"Subnet scan: found device at {found_ip}")
        return VegaConnectionInfo(
            ip=found_ip, hostname=hostname, method="subnet_scan", firmware=fw,
        )

    log.append(f"Subnet scan: no device found on :{VEGA_DEFAULT_PORT} across {total} addresses")
    return None


def _discover_direct(ip: str, port: int, log: list[str]) -> Optional[VegaConnectionInfo]:
    """Try connecting directly to a known IP:port."""
    if not ip:
        return None

    if _tcp_probe(ip, port):
        fw = _ndi_handshake(ip, port) or ""
        hostname = None
        try:
            hostname = socket.gethostbyaddr(ip)[0]
        except Exception:
            pass
        log.append(f"Direct probe: {ip}:{port} responded")
        return VegaConnectionInfo(
            ip=ip, port=port, hostname=hostname, method="direct", firmware=fw,
        )

    log.append(f"Direct probe: {ip}:{port} did not respond")
    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def discover_vega(
    known_ip: Optional[str] = "169.254.9.239",
    port: int = VEGA_DEFAULT_PORT,
    skip_subnet_scan: bool = False,
    verbose: bool = True,
) -> VegaConnectionInfo:
    """Find a Polaris Vega on the network.

    Tries strategies in order from fastest to slowest:
      1. Direct probe of known_ip (if provided)
      2. ARP cache / hostname resolution
      3. Bonjour/mDNS service browse
      4. Link-local subnet scan (can be slow; skipped if skip_subnet_scan=True)

    Returns VegaConnectionInfo on success.
    Raises VegaNotFoundError with detailed diagnostics on failure.
    """
    log: list[str] = []

    if verbose:
        print("Searching for Polaris Vega...")

    # Strategy 1: direct probe
    if known_ip:
        result = _discover_direct(known_ip, port, log)
        if result:
            if verbose:
                print(f"  Found: {result}")
            return result

    # Strategy 2: ARP cache
    result = _discover_hostname_resolve(log)
    if result:
        if verbose:
            print(f"  Found: {result}")
        return result

    # Strategy 3: Bonjour
    result = _discover_bonjour(log)
    if result:
        if verbose:
            print(f"  Found: {result}")
        return result

    # Strategy 4: subnet scan
    if not skip_subnet_scan:
        if verbose:
            print("  Bonjour/ARP failed, scanning link-local subnet (this may take a moment)...")
        result = _discover_subnet_scan(log)
        if result:
            if verbose:
                print(f"  Found: {result}")
            return result

    raise VegaNotFoundError(log)


def connect_vega(
    romfiles: list[str],
    known_ip: Optional[str] = None,
    port: int = VEGA_DEFAULT_PORT,
    skip_subnet_scan: bool = False,
    verbose: bool = True,
):
    """Discover the Vega and return a connected, tracking-ready NDITracker.

    Args:
        romfiles: List of .rom file paths to load.
        known_ip: Optional IP to try first (speeds up connection).
        port: NDI port (default 8765).
        skip_subnet_scan: Skip the slow /16 scan if faster methods fail.
        verbose: Print progress to stdout.

    Returns:
        (tracker, info) tuple — tracker is an NDITracker instance,
        info is VegaConnectionInfo.

    Raises:
        VegaNotFoundError: Camera not found.
        FileNotFoundError: A ROM file doesn't exist.
        IOError: Connection succeeded but tracker init failed.
    """
    from sksurgerynditracker.nditracker import NDITracker

    info = discover_vega(
        known_ip=known_ip, port=port,
        skip_subnet_scan=skip_subnet_scan, verbose=verbose,
    )

    settings = info.as_settings(romfiles)

    if verbose:
        print(f"  Connecting NDITracker to {info.ip}:{info.port}...")

    tracker = NDITracker(settings)

    if verbose:
        print("  Connected.")

    return tracker, info


# ---------------------------------------------------------------------------
# CLI: run as standalone script for diagnostics
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    import argparse

    parser = argparse.ArgumentParser(
        description="Discover and test connection to an NDI Polaris Vega camera.",
    )
    parser.add_argument("--ip", default=None,
                        help="Known IP to try first (e.g. 169.254.9.239)")
    parser.add_argument("--port", type=int, default=VEGA_DEFAULT_PORT,
                        help=f"NDI port (default {VEGA_DEFAULT_PORT})")
    parser.add_argument("--no-scan", action="store_true",
                        help="Skip the slow link-local subnet scan")
    args = parser.parse_args()

    try:
        info = discover_vega(
            known_ip=args.ip, port=args.port,
            skip_subnet_scan=args.no_scan, verbose=True,
        )
        print(f"\nSuccess: {info}")
        sys.exit(0)
    except VegaNotFoundError as e:
        print(f"\n{e}", file=sys.stderr)
        sys.exit(1)
