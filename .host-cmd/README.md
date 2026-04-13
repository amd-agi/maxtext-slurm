# .host-cmd

Run commands on the host from inside a Docker container, via a shared directory.

> **WARNING**: This executes arbitrary commands on the host machine with the
> privileges of whoever starts the server. Review the policy carefully
> before starting. Only run the server on machines you trust all container
> users to have host-level access to. Do NOT run this on production systems
> without proper deny/allow policies in place.

## Quick Start

### 1. On the HOST — start the server

```bash
cd /path/to/.host-cmd
./host-cmd-ctl.sh start
```

Each physical node runs its own server. The node is identified by
`HOST_CMD_NODE` (if set) or the short hostname. Start the server on every
node you want to reach from containers.

### 2. From the CONTAINER — run commands

```bash
python3 .host-cmd/host_cmd.py "hostname"
python3 .host-cmd/host_cmd.py --ping
python3 .host-cmd/host_cmd.py --status
```

Or as a library:

```python
from host_cmd import HostBridge
br = HostBridge()
result = br.run("hostname")
print(result["stdout"])
```

Inside containers, set `HOST_CMD_NODE` to the same value the host uses
(e.g. the Slurm node name) so the client targets the correct node's queue.

## Server Management

All commands run on the host via `host-cmd-ctl.sh`:

```bash
./host-cmd-ctl.sh start               # start server on this node
./host-cmd-ctl.sh stop                # stop this node's server
./host-cmd-ctl.sh restart             # restart (e.g. after policy change)
./host-cmd-ctl.sh status              # check if running
./host-cmd-ctl.sh node-id             # print this node's id
./host-cmd-ctl.sh nodes               # list all known node servers
./host-cmd-ctl.sh log                 # tail this node's server log
./host-cmd-ctl.sh history             # list recent commands and exit codes
./host-cmd-ctl.sh cleanup             # delete this node's result files
./host-cmd-ctl.sh cleanup-all         # delete all nodes' result files
./host-cmd-ctl.sh cleanup 24          # delete this node's results older than 24h
```

## Policy

Commands are filtered by deny/allow regex patterns. The server loads
the policy once at startup (immutable at runtime).

- `policy.json.default` — shipped template with sensible defaults (tracked in git)
- `policy.json` — user customization (not tracked in git)

If `policy.json` exists, it is used. Otherwise `policy.json.default` is
used. The user's file fully replaces the default (no merging).

Manage via `host-cmd-ctl.sh` (auto-restarts the server to apply):

```bash
./host-cmd-ctl.sh policy              # show current rules
./host-cmd-ctl.sh deny '\bwhoami\b'   # add a deny pattern
./host-cmd-ctl.sh allow '^squeue'     # add an allow pattern
./host-cmd-ctl.sh undeny '\bwhoami\b' # remove a deny pattern
./host-cmd-ctl.sh unallow '^squeue'   # remove an allow pattern
```

Evaluation order: deny first, then allow. If `allow_patterns` is empty,
all non-denied commands are permitted. If non-empty, only matching
commands pass.

**Security note**: `deny_patterns` is a guardrail against accidental
damage, not a security boundary. Shell commands can be obfuscated to
bypass regex filters (scripts, encoding, variable expansion, etc.).
For actual lockdown, use `allow_patterns` as a whitelist — only
explicitly matched commands will run, everything else is rejected.

## Architecture

Each physical node runs its own server process. The node id comes from
`HOST_CMD_NODE` (recommended for containers) or the short hostname.
Multiple containers on the same node share one server. Different nodes
have independent queues, locks, pid files, and logs. Results are shared.

```
                  shared directory (.host-cmd/)
                 +--------------------------------------+
Container A ---> |                                      |
Container B ---> |  queue/<node-id>/*.cmd  -------->    | ---> host_cmd_server.py
Container C ---> |  results/*.json         <--------    |      (one per node)
                 +--------------------------------------+
```

Per-node files:
- `queue/<node-id>/` — job queue (one per node)
- `running/<node-id>/` — in-flight jobs
- `daemon.<node-id>.pid` — server PID
- `daemon.<node-id>.lock` — singleton lock
- `host_cmd_server.<node-id>.log` — server log

Shared files:
- `results/` — command results (all nodes write here, keyed by UUID)
- `policy.json` — command allow/deny rules
