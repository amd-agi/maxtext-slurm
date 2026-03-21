# Notifications

A programmable notification system built on [Telegram](https://telegram.org/). Two scripts, one credential store. Part of the [Observability](observability.md) stack but works independently — no `RAY=1` required.

| Script | Purpose |
|--------|---------|
| `utils/telegram_bot.sh` | General-purpose CLI for sending and receiving Telegram messages — use it in scripts, pipelines, or interactively |
| `utils/slurm_job_monitor.sh` | Automated [Slurm](https://slurm.schedmd.com/) job monitoring with push notifications (built on `telegram_bot.sh`) |

Both scripts read credentials from `~/.tg_config` — set up once, use everywhere. Multiple bot profiles are supported for concurrent sessions or routing to different chats.

## One-time setup

Create a Telegram bot and save credentials:

1. Message [@BotFather](https://t.me/BotFather) → `/newbot` → get your **bot token**
2. Start a chat with your bot, then get your **chat ID** from `https://api.telegram.org/bot<TOKEN>/getUpdates`
3. Save credentials to `~/.tg_config`:
   ```bash
   install -m 600 /dev/null ~/.tg_config
   # Then add your credentials:
   ```
   ```
   BotToken your_token_here
   ChatID your_chat_id_here
   ```
   That's it — two lines. The file has `600` permissions (owner-only).
4. Test: `utils/telegram_bot.sh send "Hello from $(hostname)"`

Tutorial: [Telegram Bot API — Getting Started](https://core.telegram.org/bots/tutorial)

### Multiple bot profiles

Add named `Bot` blocks for concurrent sessions or different channels:

```
BotToken your_default_token
ChatID your_default_chat

Bot alerts
    BotToken your_alerts_token
    ChatID your_alerts_chat

Bot team
    BotToken your_team_token
    ChatID your_team_group_chat
    ParseMode HTML
```

Select a profile with `-b`:

```bash
utils/telegram_bot.sh -b alerts send "Job failed"
utils/telegram_bot.sh -b team send "Deploy complete"
utils/telegram_bot.sh send "Hello"  # uses default
```

Default resolution (when no `-b` is given):

1. `Bot default` block — if it exists
2. Top-level key-value pairs (before any `Bot` block)
3. First `Bot` block in the file

Each `Bot` block with a unique `BotToken` has its own Telegram message queue, allowing concurrent `recv` sessions without message conflicts.

### Legacy migration

If a legacy credential file exists from an earlier setup, `telegram_bot.sh` will automatically migrate it to `~/.tg_config` on first run (creating a `default` profile) and remove the old file. No manual action needed.

## telegram_bot.sh

A subcommand-based CLI for interacting with the Telegram Bot API. Supports `send` (send messages) and `recv` (wait for incoming messages via long polling).

### Sending messages

```bash
# Simple message (credentials from ~/.tg_config default profile)
utils/telegram_bot.sh send "Checkpoint saved at step 10000"

# Send to a named profile
utils/telegram_bot.sh -b alerts send "Job failed on $(hostname)"

# Pipe content
echo "Deploy complete on $(hostname)" | utils/telegram_bot.sh send

# Markdown formatting (default)
utils/telegram_bot.sh send "*Training complete* — final loss: 2.31"

# Plain text (no Markdown parsing) — safe for arbitrary content
PARSE_MODE="" utils/telegram_bot.sh send "file_names_with_underscores & special chars"

# Explicit credentials (override config file)
TG_BOT_TOKEN=tok TG_CHAT_ID=123 utils/telegram_bot.sh send "Hello"
```

### Receiving messages

Wait for an incoming message using Telegram's long polling. Only messages from the configured `TG_CHAT_ID` are returned. On startup, any pending messages from your chat are returned immediately; if none are pending, long-polling begins for new messages.

```bash
# Wait up to 10 minutes (default) for a message
utils/telegram_bot.sh recv

# Custom timeout (5 minutes)
utils/telegram_bot.sh recv --timeout 300
```

The received message text is printed to stdout (exit 0). If no message arrives within the timeout, exits 1. Under the hood, it long-polls the Telegram API in 60-second rounds — one held-open HTTP connection at a time, no busy polling.

```bash
# Send a question, wait for the reply
utils/telegram_bot.sh send "Should I proceed? (yes/no)"
reply=$(utils/telegram_bot.sh recv --timeout 300)
echo "User replied: $reply"
```

### Usage patterns

Compose `telegram_bot.sh` into any workflow:

```bash
# Alert on failure
train.sh || utils/telegram_bot.sh send "Training failed on $(hostname)"

# Notify on completion
train.sh && utils/telegram_bot.sh send "Training done — check results"

# In a cron job
0 */6 * * * /path/to/telegram_bot.sh send "Disk usage: $(df -h / | tail -1)"

# Wrap any long-running command
some_command; utils/telegram_bot.sh send "Command finished (exit $?)"
```

### Credential resolution

Credentials are resolved in order:

1. `TG_BOT_TOKEN` / `TG_CHAT_ID` environment variables (inline or exported)
2. `-b <name>` → named `Bot` block in `~/.tg_config`
3. Default profile from `~/.tg_config` (`Bot default` > top-level values > first `Bot` block)

### Optional environment variables

| Variable | Effect |
|----------|--------|
| `PARSE_MODE` | `Markdown` (default), `MarkdownV2`, `HTML`, or `""` for plain text |
| `DISABLE_NOTIFICATION` | `"true"` to send silently |
| `DISABLE_PREVIEW` | `"true"` to disable link previews |

## slurm_job_monitor.sh

Automated monitoring for Slurm jobs — sends push notifications for state changes, hang detection, and periodic log updates. Uses `telegram_bot.sh` under the hood.

```bash
utils/slurm_job_monitor.sh -j <slurm_job_id>
```

### What it monitors

| Notification | When |
|---|---|
| State changes | PENDING → RUNNING → COMPLETED / FAILED / CANCELLED / TIMEOUT |
| Hang alert | Log file stops updating for longer than the timeout (default 30m) |
| Resume alert | Log resumes after a hang was detected |
| Periodic updates | Last N log lines at configurable intervals (default 1h) |
| Signal handling | Graceful notification on Ctrl+C, `kill`, or SSH disconnect |

### Options

```bash
# Custom hang timeout (10 min) and update interval (30 min, last 20 lines)
utils/slurm_job_monitor.sh -j 12345 -t 600 -u 1800 -l 20

# Use a specific bot profile
utils/slurm_job_monitor.sh -j 12345 --profile alerts

# Filter periodic updates to show only errors
utils/slurm_job_monitor.sh -j 12345 -g "ERROR|WARNING"

# Exclude noisy lines from updates
utils/slurm_job_monitor.sh -j 12345 -g "DEBUG|TRACE" -v
```

### Credential resolution

Credentials are resolved in order:

1. `TG_BOT_TOKEN` / `TG_CHAT_ID` environment variables
2. `--profile <name>` → named `Bot` block in `~/.tg_config`
3. Default profile from `~/.tg_config`
4. `-b` / `-c` flags (legacy, still supported)

### Tips

- Run in [`tmux`](https://github.com/tmux/tmux) so the monitor survives SSH disconnections.
- The monitor sends a notification when interrupted by signals (Ctrl+C, `kill`, SIGHUP) — only `kill -9` bypasses this.
- Jobs with no StdOut configured are still monitored for state changes; hang detection and log updates are disabled.

---

See also: [Observability](observability.md) for the full monitoring stack (dashboards, metrics, TSDB) | [Tooling](tooling.md) for the command reference overview.
