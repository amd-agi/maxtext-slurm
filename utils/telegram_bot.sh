#!/bin/bash

# Telegram Bot CLI — a subcommand-based utility for interacting with the
# Telegram Bot API.
#
# Usage:
#   ./telegram_bot.sh [-b <bot>] <command> [args...]
#
# Commands:
#   send <message>        Send a message to a Telegram chat
#   recv [--timeout SECS] Wait for a message (long polling, default: 600s)
#
# Credentials (in order of precedence):
#    1. Environment variables: TG_BOT_TOKEN, TG_CHAT_ID
#    2. -b <name> flag → named Bot block in ~/.tg_config
#    3. Default profile from ~/.tg_config (Bot default > top-level > first block)
#    4. Auto-migrated from legacy ~/.tg_env (one-time, then deleted)
#
# Examples:
#    ./telegram_bot.sh send "Hello world"
#    ./telegram_bot.sh -b alerts send "Job failed"
#    ./telegram_bot.sh recv --timeout 300
#    TG_BOT_TOKEN=tok TG_CHAT_ID=123 ./telegram_bot.sh send "Hello world"

set -euo pipefail

readonly SCRIPT_NAME=$(basename "$0")

# ============================================================================
# COMMON HELPERS
# ============================================================================

readonly TG_CONFIG_FILE="$HOME/.tg_config"
readonly TG_ENV_FILE="$HOME/.tg_env"

# Selected bot profile (set by -b flag, empty = "default").
_TG_BOT_PROFILE=""

# Parse ~/.tg_config (SSH-config-style format) and resolve a profile.
#
# Args: profile_name
#
# When profile_name is "default", resolution order:
#   1. "Bot default" block (explicit)
#   2. Top-level key-value pairs before any Bot block (anonymous)
#   3. First Bot block in the file (implicit)
#
# When profile_name is anything else, only that exact block is matched.
#
# Sets TG_BOT_TOKEN and TG_CHAT_ID if found.
# Returns 0 on success, 1 if profile not found.
_parse_tg_config() {
    local target="$1"
    local config_file="$TG_CONFIG_FILE"

    [[ -f "$config_file" ]] || return 1

    local in_block=false target_found=false any_block_seen=false
    local token="" chat_id="" parse_mode=""
    # Fallbacks for default resolution
    local toplevel_token="" toplevel_chatid="" toplevel_parsemode=""
    local first_token="" first_chatid="" first_parsemode="" first_saved=false

    while IFS= read -r line || [[ -n "$line" ]]; do
        local stripped
        read -r stripped <<< "$line"

        [[ -z "$stripped" || "$stripped" == \#* ]] && continue

        if [[ "$stripped" =~ ^Bot[[:space:]]+(.+)$ ]]; then
            # Save the first block's data when we leave it
            if [[ "$any_block_seen" == true && "$first_saved" == false ]]; then
                first_token="$token" first_chatid="$chat_id" first_parsemode="$parse_mode"
                first_saved=true
            fi
            if [[ "$in_block" == true && "$target_found" == true ]]; then
                break
            fi
            any_block_seen=true
            token="" chat_id="" parse_mode=""
            if [[ "${BASH_REMATCH[1]}" == "$target" ]]; then
                in_block=true
                target_found=true
            else
                in_block=true
                target_found=false
            fi
            continue
        fi

        if [[ "$stripped" != *" "* ]]; then
            continue
        fi

        local key="${stripped%% *}"
        local value="${stripped#* }"

        if [[ "$any_block_seen" == false ]]; then
            # Top-level key-value pairs (before any Bot block)
            case "$key" in
                BotToken)  toplevel_token="$value" ;;
                ChatID)    toplevel_chatid="$value" ;;
                ParseMode) toplevel_parsemode="$value" ;;
            esac
        elif [[ "$in_block" == true ]]; then
            case "$key" in
                BotToken)  token="$value" ;;
                ChatID)    chat_id="$value" ;;
                ParseMode) parse_mode="$value" ;;
            esac
        fi
    done < "$config_file"

    # Save first block if we only had one block in the file
    if [[ "$any_block_seen" == true && "$first_saved" == false ]]; then
        first_token="$token" first_chatid="$chat_id" first_parsemode="$parse_mode"
        first_saved=true
    fi

    # Select which values to apply
    local final_token="" final_chatid="" final_parsemode=""

    if [[ "$target_found" == true ]]; then
        final_token="$token" final_chatid="$chat_id" final_parsemode="$parse_mode"
    elif [[ "$target" == "default" ]]; then
        # Fallback: top-level values, then first block
        if [[ -n "$toplevel_token" || -n "$toplevel_chatid" ]]; then
            final_token="$toplevel_token" final_chatid="$toplevel_chatid" final_parsemode="$toplevel_parsemode"
        elif [[ "$first_saved" == true ]]; then
            final_token="$first_token" final_chatid="$first_chatid" final_parsemode="$first_parsemode"
        else
            return 1
        fi
    else
        return 1
    fi

    [[ -n "$final_token" ]]   && TG_BOT_TOKEN="$final_token"
    [[ -n "$final_chatid" ]]  && TG_CHAT_ID="$final_chatid"
    [[ -n "$final_parsemode" && -z "${PARSE_MODE+x}" ]] && PARSE_MODE="$final_parsemode"
    return 0
}

# Migrate legacy ~/.tg_env to ~/.tg_config and delete the old file.
# Only runs when ~/.tg_config does not exist and ~/.tg_env does.
_migrate_tg_env() {
    [[ ! -f "$TG_CONFIG_FILE" && -f "$TG_ENV_FILE" ]] || return 1

    echo "Migrating $TG_ENV_FILE → $TG_CONFIG_FILE ..." >&2

    local token="" chat_id=""

    # Extract values without sourcing (safer — avoids executing arbitrary code)
    while IFS= read -r line || [[ -n "$line" ]]; do
        [[ -z "$line" || "$line" =~ ^[[:space:]]*# ]] && continue
        if [[ "$line" =~ TG_BOT_TOKEN=[\"\']*([^\"\']+) ]]; then
            token="${BASH_REMATCH[1]}"
        elif [[ "$line" =~ TG_CHAT_ID=[\"\']*([^\"\']+) ]]; then
            chat_id="${BASH_REMATCH[1]}"
        fi
    done < "$TG_ENV_FILE"

    if [[ -z "$token" || -z "$chat_id" ]]; then
        echo "Warning: could not parse $TG_ENV_FILE — skipping migration" >&2
        return 1
    fi

    install -m 600 /dev/null "$TG_CONFIG_FILE"
    cat > "$TG_CONFIG_FILE" << EOF
BotToken $token
ChatID $chat_id
EOF

    if [[ -f "$TG_CONFIG_FILE" ]] && _parse_tg_config "default" > /dev/null 2>&1; then
        rm -f "$TG_ENV_FILE"
        echo "Migration complete. $TG_ENV_FILE removed." >&2
        return 0
    else
        echo "Warning: migration verification failed — keeping $TG_ENV_FILE" >&2
        rm -f "$TG_CONFIG_FILE"
        return 1
    fi
}

# Load credentials from the config chain.
# Resolution order:
#   1. Inline env vars (TG_BOT_TOKEN / TG_CHAT_ID already set)
#   2. -b <name> → named Bot block in ~/.tg_config
#   3. Default profile from ~/.tg_config (Bot default > top-level > first block)
#   4. Auto-migrate from ~/.tg_env (one-time)
load_env() {
    # Already have both credentials from env vars — done.
    if [[ -n "${TG_BOT_TOKEN:-}" && -n "${TG_CHAT_ID:-}" ]]; then
        return
    fi

    local profile="${_TG_BOT_PROFILE:-default}"

    # Try ~/.tg_config
    if _parse_tg_config "$profile"; then
        echo "Using Bot profile '$profile' from $TG_CONFIG_FILE" >&2
        return
    fi

    # If a non-default profile was requested but not found, that's an error.
    if [[ "$profile" != "default" ]]; then
        echo "Error: Bot profile '$profile' not found in $TG_CONFIG_FILE" >&2
        exit 1
    fi

    # Try auto-migration from legacy ~/.tg_env
    if _migrate_tg_env; then
        _parse_tg_config "default" 2>/dev/null && return
    fi

    # Last resort: source ~/.tg_env directly (migration failed but file exists)
    if [[ -f "$TG_ENV_FILE" ]]; then
        echo "Sourcing credentials from $TG_ENV_FILE" >&2
        # shellcheck source=/dev/null
        source "$TG_ENV_FILE"
    fi
}

# Validate that required environment variables are set.
validate_env() {
    load_env

    if [[ -z "${TG_BOT_TOKEN:-}" ]]; then
        echo "Error: TG_BOT_TOKEN not set. Create $TG_CONFIG_FILE or export it." >&2
        echo "See docs/notifications.md for setup instructions." >&2
        exit 1
    fi
    if [[ -z "${TG_CHAT_ID:-}" ]]; then
        echo "Error: TG_CHAT_ID not set. Create $TG_CONFIG_FILE or export it." >&2
        echo "See docs/notifications.md for setup instructions." >&2
        exit 1
    fi
}

# ============================================================================
# COMMAND: send
# ============================================================================

# Perform a single sendMessage API call.
# Args: message, parse_mode (may be empty for plain text)
# Prints the API response (or curl error) to stdout.
_do_send() {
    local message="$1"
    local parse_mode="$2"
    local api_url="https://api.telegram.org/bot${TG_BOT_TOKEN}/sendMessage"

    local curl_args=(
        -sS
        -X POST
        "$api_url"
        --data-urlencode "chat_id=${TG_CHAT_ID}"
        --data-urlencode "text=${message}"
    )

    if [[ -n "$parse_mode" ]]; then
        curl_args+=(--data-urlencode "parse_mode=${parse_mode}")
    fi

    if [[ "${DISABLE_NOTIFICATION:-}" == "true" ]]; then
        curl_args+=(--data-urlencode "disable_notification=true")
    fi

    if [[ "${DISABLE_PREVIEW:-}" == "true" ]]; then
        curl_args+=(--data-urlencode "disable_web_page_preview=true")
    fi

    curl "${curl_args[@]}" 2>&1
}

# Split a message into Telegram-safe chunks (max 4096 chars each).
# Tracks Markdown ``` code blocks: closes them at chunk boundaries
# and reopens in the next chunk so formatting is preserved.
# Outputs NUL-delimited chunks for safe reading with `read -d ''`.
_split_message() {
    local message="$1"
    local max_len=4096
    # Reserve space for code fence close/open and [n/m] part prefix
    local overhead=40
    local effective=$((max_len - overhead))

    # Fast path: message already fits
    if [[ ${#message} -le $max_len ]]; then
        printf '%s\0' "$message"
        return
    fi

    local chunk=""
    local in_code_block=false

    while IFS= read -r line || [[ -n "$line" ]]; do
        local candidate
        if [[ -z "$chunk" ]]; then
            candidate="$line"
        else
            candidate="${chunk}"$'\n'"${line}"
        fi

        if [[ ${#candidate} -gt $effective && -n "$chunk" ]]; then
            # Adding this line would exceed the limit — flush current chunk
            if [[ "$in_code_block" == true ]]; then
                chunk+=$'\n```'
            fi
            printf '%s\0' "$chunk"

            # Start next chunk, reopening code block if we were inside one
            if [[ "$in_code_block" == true ]]; then
                chunk=$'```\n'"$line"
            else
                chunk="$line"
            fi
        else
            chunk="$candidate"
        fi

        # Track code block state: odd number of ``` on a line toggles the flag
        local tmp="$line"
        local fence_count=0
        while [[ "$tmp" == *'```'* ]]; do
            fence_count=$((fence_count + 1))
            tmp="${tmp#*\`\`\`}"
        done
        if [[ $((fence_count % 2)) -eq 1 ]]; then
            if [[ "$in_code_block" == true ]]; then
                in_code_block=false
            else
                in_code_block=true
            fi
        fi
    done <<< "$message"

    # Flush remaining content
    if [[ -n "$chunk" ]]; then
        printf '%s\0' "$chunk"
    fi
}

# Send a message to Telegram.
#
# Message can come from:
#   1. Command-line arguments (joined with spaces)
#   2. Standard input (piped or redirected)
#
# Extra environment variables (optional):
#   PARSE_MODE             Markdown (default), MarkdownV2, HTML, or "" for plain text
#   DISABLE_NOTIFICATION   "true" to send silently
#   DISABLE_PREVIEW        "true" to disable link previews
cmd_send() {
    # --- read message ---
    local message=""

    if [[ $# -gt 0 ]]; then
        message="$*"
    elif [[ ! -t 0 ]]; then
        message=$(cat)
    else
        cat >&2 << EOF
Usage: $SCRIPT_NAME [-b <bot>] send <message>
    or: echo 'message' | $SCRIPT_NAME [-b <bot>] send
    or: $SCRIPT_NAME [-b <bot>] send <<< 'message'

Credentials (in order of precedence):
    1. TG_BOT_TOKEN / TG_CHAT_ID environment variables
    2. -b <name> → named Bot block in ~/.tg_config
    3. Default profile from ~/.tg_config

Optional environment variables:
    PARSE_MODE             Markdown (default), MarkdownV2, HTML, or "" for plain text
    DISABLE_NOTIFICATION   "true" to send silently
    DISABLE_PREVIEW        "true" to disable link previews
EOF
        exit 1
    fi

    if [[ -z "$message" ]]; then
        echo "Error: Message is empty" >&2
        exit 1
    fi

    validate_env

    # --- split into Telegram-safe chunks and send with plain-text fallback ---
    local parse_mode="${PARSE_MODE-Markdown}"

    local -a chunks=()
    while IFS= read -r -d '' chunk; do
        chunks+=("$chunk")
    done < <(_split_message "$message")

    local total=${#chunks[@]}

    if [[ $total -eq 0 ]]; then
        echo "Error: Internal error — message produced no chunks" >&2
        exit 1
    fi

    for i in "${!chunks[@]}"; do
        local part=$((i + 1))
        local chunk="${chunks[$i]}"

        # Add part indicator for multi-part messages
        if [[ $total -gt 1 ]]; then
            chunk="[${part}/${total}] ${chunk}"
        fi

        local response
        response=$(_do_send "$chunk" "$parse_mode")

        if echo "$response" | grep -q '"ok":true'; then
            continue
        fi

        # Retry this chunk as plain text
        if [[ -n "$parse_mode" ]]; then
            echo "Warning: parse_mode=${parse_mode} failed on part ${part}/${total}, retrying as plain text..." >&2
            response=$(_do_send "$chunk" "")
            if echo "$response" | grep -q '"ok":true'; then
                continue
            fi
        fi

        echo "Error: Telegram API request failed (part ${part}/${total})" >&2
        echo "$response" >&2
        exit 1
    done

    exit 0
}

# ============================================================================
# COMMAND: recv
# ============================================================================

# Parse a getUpdates API response, extracting ALL messages matching our chat.
# Reads JSON from stdin.
# Args: chat_id
# On match: prints "max_update_id\n<all message texts joined by newlines>"
#           to stdout, exits 0.
# API error: prints diagnostic to stderr, exits 2.
# No match: exits 1.
_parse_recv_response() {
    local chat_id="$1"
    python3 -c "
import json, sys
data = json.load(sys.stdin)
if not data.get('ok'):
    print('API error: ' + json.dumps(data), file=sys.stderr)
    sys.exit(2)
chat_id = int(sys.argv[1])
max_id = None
texts = []
for r in data.get('result', []):
    msg = r.get('message') or r.get('edited_message') or {}
    if msg.get('chat', {}).get('id') == chat_id:
        text = msg.get('text', '')
        if text:
            max_id = r['update_id']
            texts.append(text)
if texts:
    print(max_id)
    print('\n'.join(texts))
    sys.exit(0)
sys.exit(1)
" "$chat_id"
}

# Wait for a message from Telegram using long polling.
#
# Usage: cmd_recv [--timeout SECONDS]
#
# Prints the received message text to stdout and exits 0.
# Exits 1 if no message is received within the timeout.
#
# Default timeout: 600 seconds (10 minutes).
# Uses Telegram's long polling (60s per round) to minimize HTTP requests.
cmd_recv() {
    local total_timeout=600
    local poll_interval=60

    while [[ $# -gt 0 ]]; do
        case "$1" in
            --timeout)
                total_timeout="$2"
                shift 2
                ;;
            --timeout=*)
                total_timeout="${1#--timeout=}"
                shift
                ;;
            -h|--help)
                echo "Usage: $SCRIPT_NAME recv [--timeout SECONDS]" >&2
                echo "Wait for a message from Telegram (default timeout: 600s)." >&2
                exit 0
                ;;
            *)
                echo "Error: Unknown option: $1" >&2
                echo "Usage: $SCRIPT_NAME recv [--timeout SECONDS]" >&2
                exit 1
                ;;
        esac
    done

    validate_env

    local api_url="https://api.telegram.org/bot${TG_BOT_TOKEN}/getUpdates"

    # Phase 1: Check for pending messages (non-blocking).
    # This catches messages sent while the agent was busy working.
    local pending_response
    pending_response=$(curl -sS --max-time 10 "${api_url}?timeout=0" 2>&1)

    local pending_parsed="" pending_exit=0
    pending_parsed=$(echo "$pending_response" | _parse_recv_response "$TG_CHAT_ID") || pending_exit=$?

    if [[ $pending_exit -eq 0 ]]; then
        local update_id msg_text
        update_id=$(head -1 <<< "$pending_parsed")
        msg_text=$(tail -n +2 <<< "$pending_parsed")

        curl -sS --max-time 10 \
            "${api_url}?offset=$((update_id + 1))&timeout=0" > /dev/null 2>&1 || true

        echo "$msg_text"
        exit 0
    fi

    # Phase 2: No pending messages from our chat. Set offset past all pending
    # updates so we only long-poll for genuinely new messages.
    local next_offset=0
    local latest_id
    latest_id=$(echo "$pending_response" | python3 -c "
import json, sys
data = json.load(sys.stdin)
results = data.get('result', [])
if results:
    print(max(r['update_id'] for r in results))
else:
    print('')
" 2>/dev/null) || true

    if [[ -n "$latest_id" ]]; then
        next_offset=$((latest_id + 1))
    fi

    local start_time
    start_time=$(date +%s)

    echo "Waiting for message (timeout: ${total_timeout}s)..." >&2

    while true; do
        local now elapsed remaining
        now=$(date +%s)
        elapsed=$((now - start_time))

        if [[ $elapsed -ge $total_timeout ]]; then
            echo "Timeout: no message received within ${total_timeout}s" >&2
            exit 1
        fi

        remaining=$((total_timeout - elapsed))
        local this_poll=$poll_interval
        if [[ $remaining -lt $poll_interval ]]; then
            this_poll=$remaining
        fi
        local curl_timeout=$((this_poll + 10))

        local response
        response=$(curl -sS --max-time "$curl_timeout" \
            "${api_url}?offset=${next_offset}&timeout=${this_poll}" 2>&1) || {
            echo "Warning: curl failed, retrying..." >&2
            sleep 2
            continue
        }

        local parsed="" parse_exit=0
        parsed=$(echo "$response" | _parse_recv_response "$TG_CHAT_ID") || parse_exit=$?

        if [[ $parse_exit -eq 0 ]]; then
            local update_id msg_text
            update_id=$(head -1 <<< "$parsed")
            msg_text=$(tail -n +2 <<< "$parsed")

            # Confirm the processed update
            curl -sS --max-time 10 \
                "${api_url}?offset=$((update_id + 1))&timeout=0" > /dev/null 2>&1 || true

            echo "$msg_text"
            exit 0
        elif [[ $parse_exit -eq 2 ]]; then
            echo "Warning: API error, retrying..." >&2
            sleep 2
        fi
    done
}

# ============================================================================
# MAIN DISPATCH
# ============================================================================

usage() {
    cat >&2 << EOF
Usage: $SCRIPT_NAME [-b <bot>] <command> [args...]

Commands:
    send <message>          Send a message to a Telegram chat
    recv [--timeout SECS]   Wait for a message (default: 600s)

Options:
    -b <name>   Select a Bot profile from ~/.tg_config (default: "default")

Credentials (in order of precedence):
    1. TG_BOT_TOKEN / TG_CHAT_ID environment variables
    2. -b <name> → named Bot block in ~/.tg_config
    3. Default profile from ~/.tg_config

Examples:
    $SCRIPT_NAME send "Hello world"
    $SCRIPT_NAME -b alerts send "Job failed"
    $SCRIPT_NAME recv --timeout 300
    TG_BOT_TOKEN=tok TG_CHAT_ID=123 $SCRIPT_NAME send "Hello world"
    echo "message" | $SCRIPT_NAME send

Config file (~/.tg_config):
    BotToken <token>
    ChatID <chat_id>

    Bot alerts
        BotToken <token>
        ChatID <chat_id>

Setup: See docs/notifications.md for one-time Telegram bot creation and config setup.
EOF
    exit 1
}

if [[ $# -lt 1 ]]; then
    usage
fi

# Parse global flags before the subcommand.
while [[ $# -gt 0 ]]; do
    case "$1" in
        -b)
            [[ $# -ge 2 ]] || { echo "Error: -b requires a profile name" >&2; exit 1; }
            _TG_BOT_PROFILE="$2"
            shift 2
            ;;
        -b=*)
            _TG_BOT_PROFILE="${1#-b=}"
            shift
            ;;
        -h|--help|help)
            usage
            ;;
        *)
            break
            ;;
    esac
done

if [[ $# -lt 1 ]]; then
    usage
fi

COMMAND="$1"
shift

case "$COMMAND" in
    send)  cmd_send "$@" ;;
    recv)  cmd_recv "$@" ;;
    _resolve_env)
        # Internal: resolve credentials and print them for eval by callers.
        validate_env
        printf "TG_BOT_TOKEN=%q; TG_CHAT_ID=%q\n" "$TG_BOT_TOKEN" "$TG_CHAT_ID"
        ;;
    -h|--help|help) usage ;;
    *)
        echo "Error: Unknown command: $COMMAND" >&2
        usage
        ;;
esac
