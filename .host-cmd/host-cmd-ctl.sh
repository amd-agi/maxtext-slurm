#!/usr/bin/env bash
#
# host-cmd-ctl.sh — manage the per-node host-cmd server.
# Run ON THE HOST (not inside a container).
#
# Each physical node runs its own server instance. The node is identified
# by HOST_CMD_NODE (if set) or the short hostname. All state lives in
# .host-cmd/ on shared storage so containers can submit jobs.
#
# Usage:
#   ./host-cmd-ctl.sh start                 # start server on this node
#   ./host-cmd-ctl.sh stop                  # stop this node's server
#   ./host-cmd-ctl.sh restart               # restart this node's server
#   ./host-cmd-ctl.sh status                # is this node's server running?
#   ./host-cmd-ctl.sh node-id               # print this node's id
#   ./host-cmd-ctl.sh nodes                 # list all known node servers
#   ./host-cmd-ctl.sh log [N]               # tail this node's log (default 50)
#   ./host-cmd-ctl.sh history [N]           # list recent command results
#   ./host-cmd-ctl.sh cleanup [HOURS]       # delete this node's results
#   ./host-cmd-ctl.sh cleanup-all [HOURS]   # delete ALL nodes' results
#   ./host-cmd-ctl.sh policy                # show current policy
#   ./host-cmd-ctl.sh deny PATTERN          # add a deny pattern (restarts server)
#   ./host-cmd-ctl.sh allow PATTERN         # add an allow pattern (restarts server)
#   ./host-cmd-ctl.sh undeny PATTERN        # remove a deny pattern (restarts server)
#   ./host-cmd-ctl.sh unallow PATTERN       # remove an allow pattern (restarts server)
#
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NODE_ID="$(python3 "$DIR/host_cmd_common.py" node-id)"
PID_FILE="$DIR/daemon.${NODE_ID}.pid"
LOG_FILE="$DIR/host_cmd_server.${NODE_ID}.log"
SERVER="$DIR/host_cmd_server.py"
CLIENT="$DIR/host_cmd.py"

# Legacy (pre-per-node) pid / lock files — used for migration.
LEGACY_PID_FILE="$DIR/daemon.pid"
LEGACY_LOCK_FILE="$DIR/daemon.lock"

# ── helpers ───────────────────────────────────────────────────────────────

_pid_from_file() {
    # Print the PID from a file if the file exists and the process is alive.
    local f="$1"
    [ -f "$f" ] || return 1
    local pid
    pid=$(cat "$f" 2>/dev/null) || return 1
    [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null && echo "$pid" && return 0
    return 1
}

is_running() {
    _pid_from_file "$PID_FILE" >/dev/null 2>&1
}

_legacy_pid() {
    # Print the live legacy PID — checks daemon.pid first, then daemon.lock.
    local pid
    pid=$(_pid_from_file "$LEGACY_PID_FILE") && echo "$pid" && return 0
    pid=$(_pid_from_file "$LEGACY_LOCK_FILE") && echo "$pid" && return 0
    return 1
}

legacy_is_running() {
    _legacy_pid >/dev/null 2>&1
}

_stop_pid() {
    # Stop a process by PID, cleaning up the given pid file.
    local pid="$1" pidfile="$2" label="$3"
    echo "Stopping $label (PID $pid) ..."
    kill "$pid" 2>/dev/null || true
    for _ in $(seq 1 10); do
        if ! kill -0 "$pid" 2>/dev/null; then
            echo "Stopped"
            rm -f "$pidfile"
            return 0
        fi
        sleep 0.5
    done
    echo "Force killing ..."
    kill -9 "$pid" 2>/dev/null || true
    rm -f "$pidfile"
    echo "Stopped"
}

_migrate_legacy() {
    # Detect and stop a legacy (flat-layout) server on this node.
    # Returns 0 if a legacy server was found and handled, 1 otherwise.
    local lpid
    lpid=$(_legacy_pid) || return 1
    echo "Found legacy server (PID $lpid) using old layout."
    _stop_pid "$lpid" "$LEGACY_PID_FILE" "legacy server"
    rm -f "$LEGACY_LOCK_FILE"
    return 0
}

# ── commands ──────────────────────────────────────────────────────────────

cmd_start() {
    if is_running; then
        echo "Already running on $NODE_ID (PID $(cat "$PID_FILE"))"
        return 0
    fi
    _migrate_legacy || true
    echo "Starting host-cmd server on $NODE_ID ..."
    nohup python3 "$SERVER" > /dev/null 2>&1 &
    sleep 1
    if is_running; then
        echo "Started on $NODE_ID (PID $(cat "$PID_FILE"))"
        echo "Log: $LOG_FILE"
    else
        echo "Failed to start on $NODE_ID. Check:"
        echo "  $LOG_FILE"
        return 1
    fi
}

cmd_stop() {
    if is_running; then
        local pid
        pid=$(cat "$PID_FILE")
        _stop_pid "$pid" "$PID_FILE" "$NODE_ID"
        return
    fi
    if legacy_is_running; then
        local lpid
        lpid=$(_legacy_pid)
        echo "No per-node server, but found legacy server (PID $lpid)."
        _stop_pid "$lpid" "$LEGACY_PID_FILE" "legacy on $NODE_ID"
        rm -f "$LEGACY_LOCK_FILE"
        return
    fi
    echo "Not running on $NODE_ID"
}

cmd_status() {
    echo "Node id: $NODE_ID"
    if is_running; then
        echo "Server:  RUNNING (PID $(cat "$PID_FILE"))"
        echo "Log:     $LOG_FILE"
    elif legacy_is_running; then
        local lpid
        lpid=$(_legacy_pid)
        echo "Server:  RUNNING (legacy layout, PID $lpid)"
        echo "         Run '$0 restart' to migrate to per-node layout."
    else
        echo "Server:  NOT RUNNING"
        if [ -f "$PID_FILE" ]; then
            echo "         (stale pid file: $PID_FILE)"
        fi
    fi
    local qdir="$DIR/queue/$NODE_ID"
    local rdir="$DIR/running/$NODE_ID"
    local pending=0 running=0
    [ -d "$qdir" ]   && pending=$(find "$qdir" -maxdepth 1 -name '*.cmd' 2>/dev/null | wc -l)
    [ -d "$rdir" ]   && running=$(find "$rdir" -maxdepth 1 -name '*.cmd' 2>/dev/null | wc -l)
    echo "Pending: $pending  Running: $running"
}

cmd_node_id() {
    echo "$NODE_ID"
}

cmd_nodes() {
    local found=0
    # Per-node pid files: daemon.<node-id>.pid
    for pf in "$DIR"/daemon.*.pid; do
        [ -f "$pf" ] || continue
        found=1
        local nid pid state
        nid="${pf#"$DIR"/daemon.}"
        nid="${nid%.pid}"
        pid=$(cat "$pf" 2>/dev/null || echo "?")
        if kill -0 "$pid" 2>/dev/null; then
            state="RUNNING"
        else
            state="DEAD (stale pid $pid)"
        fi
        local logf="$DIR/host_cmd_server.${nid}.log"
        local last_log=""
        if [ -f "$logf" ]; then
            last_log="  $(tail -1 "$logf" 2>/dev/null)"
        fi
        printf "%-20s  PID %-8s  %s%s\n" "$nid" "$pid" "$state" "$last_log"
    done
    # Legacy flat layout: daemon.pid / daemon.lock (not yet migrated).
    if legacy_is_running; then
        found=1
        local lpid
        lpid=$(_legacy_pid)
        local last_log=""
        if [ -f "$DIR/host_cmd_server.log" ]; then
            last_log="  $(tail -1 "$DIR/host_cmd_server.log" 2>/dev/null)"
        fi
        printf "%-20s  PID %-8s  %s%s\n" "(legacy)" "$lpid" "RUNNING — restart to migrate" "$last_log"
    fi
    if [ "$found" -eq 0 ]; then
        echo "No node servers found."
    fi
}

cmd_log() {
    local n="${1:-50}"
    local f="$LOG_FILE"
    if [ ! -f "$f" ]; then
        # Fall back to the legacy (pre-per-node) log.
        f="$DIR/host_cmd_server.log"
    fi
    if [ ! -f "$f" ]; then
        echo "No log file for $NODE_ID yet."
        return 1
    fi
    echo "=== $f (last $n lines) ==="
    tail -n "$n" "$f"
}

cmd_history() {
    local limit="${1:-20}"
    local results_dir="$DIR/results"
    if [ ! -d "$results_dir" ] || [ -z "$(ls "$results_dir"/*.json 2>/dev/null)" ]; then
        echo "No results"
        return
    fi
    ls -t "$results_dir"/*.json | head -n "$limit" | while read -r f; do
        local ts cmd exit_code
        ts=$(python3 -c "import json; d=json.load(open('$f')); print(d.get('started_at','?'))" 2>/dev/null)
        cmd=$(python3 -c "import json; d=json.load(open('$f')); print(d.get('command','?')[:60])" 2>/dev/null)
        exit_code=$(python3 -c "import json; d=json.load(open('$f')); print(d.get('exit_code','?'))" 2>/dev/null)
        local date_str
        date_str=$(python3 -c "import time; print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime($ts)))" 2>/dev/null || echo "?")
        printf "%-20s  exit=%-4s  %s\n" "$date_str" "$exit_code" "$cmd"
    done
}

cmd_cleanup() {
    python3 "$CLIENT" --cleanup "$@"
}

cmd_cleanup_all() {
    python3 "$CLIENT" --cleanup "$@" --all-nodes
}

cmd_policy() {
    python3 "$CLIENT" --policy
}

cmd_deny() {
    [ -z "${1:-}" ] && { echo "Usage: $0 deny PATTERN"; exit 1; }
    python3 "$CLIENT" --deny "$1"
    echo "Restarting server to apply ..."
    cmd_stop
    cmd_start
}

cmd_allow() {
    [ -z "${1:-}" ] && { echo "Usage: $0 allow PATTERN"; exit 1; }
    python3 "$CLIENT" --allow "$1"
    echo "Restarting server to apply ..."
    cmd_stop
    cmd_start
}

cmd_undeny() {
    [ -z "${1:-}" ] && { echo "Usage: $0 undeny PATTERN"; exit 1; }
    python3 "$CLIENT" --undeny "$1"
    echo "Restarting server to apply ..."
    cmd_stop
    cmd_start
}

cmd_unallow() {
    [ -z "${1:-}" ] && { echo "Usage: $0 unallow PATTERN"; exit 1; }
    python3 "$CLIENT" --unallow "$1"
    echo "Restarting server to apply ..."
    cmd_stop
    cmd_start
}

cmd_help() {
    cat <<'EOF'
host-cmd-ctl.sh — per-node host-cmd server management

  start               Start the server on this node
  stop                Stop this node's server
  restart             Restart this node's server
  status              Show this node's server status and queue counts
  node-id             Print this node's id (from HOST_CMD_NODE or hostname)
  nodes               List all known node servers and their state
  log [N]             Show the last N lines of this node's log (default 50)
  history [N]         Show the last N command results (default 20)
  cleanup [HOURS]     Delete this node's result files (all, or older than HOURS)
  cleanup-all [HOURS] Delete ALL nodes' result files
  policy              Show the current command policy
  deny PATTERN        Add a deny regex (restarts server)
  allow PATTERN       Add an allow regex (restarts server)
  undeny PATTERN      Remove a deny regex (restarts server)
  unallow PATTERN     Remove an allow regex (restarts server)

Node identity:
  The server node id is resolved from HOST_CMD_NODE (if set), otherwise the
  short hostname. Each node gets its own queue, lock, pid, and log file.
  Set HOST_CMD_NODE the same way on the host and inside containers so they
  agree on which queue to use.

EOF
}

# ── dispatch ──────────────────────────────────────────────────────────────

case "${1:-}" in
    start)    cmd_start ;;
    stop)     cmd_stop ;;
    restart)  cmd_stop; cmd_start ;;
    status)   cmd_status ;;
    node-id)  cmd_node_id ;;
    nodes)    cmd_nodes ;;
    log)      cmd_log "${2:-50}" ;;
    history)  cmd_history "${2:-20}" ;;
    cleanup)  shift; cmd_cleanup "$@" ;;
    cleanup-all) shift; cmd_cleanup_all "$@" ;;
    policy)   cmd_policy ;;
    deny)     cmd_deny "${2:-}" ;;
    allow)    cmd_allow "${2:-}" ;;
    undeny)   cmd_undeny "${2:-}" ;;
    unallow)  cmd_unallow "${2:-}" ;;
    help|-h|--help) cmd_help ;;
    *)
        cmd_help >&2
        exit 1
        ;;
esac
