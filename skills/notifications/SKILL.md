---
name: notifications
description: Send and receive Telegram messages. Use when the user asks to be notified, messaged, or alerted about task completion, job status, or any result — and when they want to reply with follow-up instructions via Telegram. This is a cross-cutting skill — other skills (batch-sweep, model-config, job-triage) can use it when the user explicitly requests notification.
---

# Notifications

Send and receive Telegram messages via `utils/telegram_bot.sh`. Use when the user says things like "send me a TG message", "notify me when done", "alert me on Telegram", or "wait for my TG reply".

## Sending flow

**NEVER read, print, or log the contents of `~/.tg_config`.** Only check if the file exists. The script loads credentials internally.

If the user specifies a bot profile (e.g., "use the alerts channel"), pass `-b <name>` on every `send`/`recv` call for the rest of the session. The choice is sticky — do not revert to default until the user explicitly asks to switch.

Try in order. Stop at the first success.

### Step 1: Try from the container

```bash
test -f ~/.tg_config && echo "EXISTS" || echo "NOT FOUND"
```

If `~/.tg_config` exists, send directly:

```bash
utils/telegram_bot.sh send "Your message here"
utils/telegram_bot.sh -b alerts send "Your message here"  # named profile
```

For multi-line messages, pipe from stdin:

```bash
echo "Line 1
Line 2" | utils/telegram_bot.sh send
```

If this succeeds, done. If `~/.tg_config` doesn't exist, go to Step 2.

### Step 2: Try from the host via host-cmd

Check if host-cmd is available:

```bash
python3 /maxtext-slurm/.host-cmd/host_cmd.py --ping --timeout 5
```

If it does NOT respond `ALIVE`, go to Step 3.

If alive, check host credentials exist (never read or print the file contents):

```bash
python3 /maxtext-slurm/.host-cmd/host_cmd.py --timeout 10 "test -f ~/.tg_config && echo EXISTS || echo NOT_FOUND"
```

If credentials exist, send using the **write-to-file pattern** (direct quoting through host-cmd breaks on special characters):

```bash
# Write message to a temp file on the host
python3 /maxtext-slurm/.host-cmd/host_cmd.py --timeout 10 "cat > /tmp/tg_msg.txt << 'EOF'
Your message here.
Multiple lines are fine.
Special chars & (parens) work.
EOF"

# Pipe the file into telegram_bot.sh
python3 /maxtext-slurm/.host-cmd/host_cmd.py --timeout 15 \
  "cat /tmp/tg_msg.txt | bash utils/telegram_bot.sh send"
```

To use a named profile via host-cmd: `bash utils/telegram_bot.sh -b alerts send`

If this succeeds, done. If credentials don't exist on the host either, go to Step 3.

### Step 3: Report failure and offer help

Tell the user what failed and give minimal next steps. Example:

> Could not send Telegram notification — no credentials found.
> `~/.tg_config` is missing in both the container and the host.
> Want me to help set up a Telegram bot? (takes ~2 minutes)

Or if host-cmd is unavailable:

> Could not send Telegram notification — `~/.tg_config` not found in the container, and host-cmd is not available.
> Want me to help set up Telegram credentials locally?

If the user says yes, walk them through `docs/notifications.md` setup:

1. Message @BotFather on Telegram → `/newbot` → get **bot token**
2. Start a chat with the bot, get **chat ID** from `https://api.telegram.org/bot<TOKEN>/getUpdates`
3. Create `~/.tg_config` with `install -m 600 /dev/null ~/.tg_config`, then add:
   ```
   BotToken <token>
   ChatID <chat_id>
   ```
4. Test: `utils/telegram_bot.sh send "Hello from $(hostname)"`

## Host-cmd pitfalls

| Problem | Cause | Fix |
|---------|-------|-----|
| `syntax error near unexpected token` | Special chars in inline message | Use the write-to-file pattern above |
| `command not found` on host | `utils/` path is relative to repo root | host-cmd cwd is already the repo root; `bash utils/telegram_bot.sh` works |
| Empty message error | Heredoc EOF marker was indented | Use unindented `EOF` marker |

## Message formatting

Keep messages concise. Telegram has a 4096-char limit per message (the script auto-splits longer messages). Use Markdown formatting — the script sends with `PARSE_MODE=Markdown` by default.

**Wrap technical identifiers in backtick code spans** — file paths, config names, variable names, model names. Telegram does not parse Markdown inside backticks, so underscores and brackets are safe:

```
*Sweep complete*

`config_llama_70b_batch_8`: 1250 TGS
`config_llama_70b_batch_16`: 892 TGS
_Best result_: `per_device_batch_size`=4
```

This is the key rule for avoiding Markdown conflicts. `*bold*`, `_italic_`, and `` `code` `` all work naturally as long as dynamic content with underscores goes in backticks.

If Markdown still fails (e.g., unmatched `_` the agent missed), the script automatically retries as plain text — the message always gets delivered.

For multi-line code or shell output, use triple-backtick code blocks:

~~~
```
step 100: loss=2.31, TGS=1250.3
step 200: loss=2.15, TGS=1248.7
```
~~~

## Interactive loop

**Every time you send a TG notification, enter an interactive wait loop.** This lets the user reply on Telegram with follow-up instructions without returning to the Cursor GUI. All rules in this section (recv, clarification, multi-message handling, timeout) apply only while the loop is active — do NOT call recv or wait for TG replies outside this loop.

### Protocol

1. **Send the result**, then **send a separate hint message**. The result uses Markdown (wrap technical identifiers in backticks — see Message formatting above). The hint is its own message:

   > ━━━━━━━━━━━━━━━━━━━━
   > 💬 \*Awaiting further instructions\*
   > ⏳ \_Timeout: {duration}\_
   > ━━━━━━━━━━━━━━━━━━━━

Replace `{duration}` with the actual timeout in the most natural unit (e.g., "10 minutes", "1 hour", "2 hours"). The `*...*` renders as **bold** and `_..._` renders as _italic_ in Telegram. The ━ line, 💬, and ⏳ are literal characters. Send this hint after every result notification that enters the recv loop — NOT after echo messages (step 3) or progress reports (step 8).

2. **Run `recv`** to wait for the user's reply. Background it immediately so you can poll:

From the container:

```bash
utils/telegram_bot.sh recv --timeout 600
```

Via host-cmd (set host-cmd timeout slightly above recv timeout):

```bash
python3 /maxtext-slurm/.host-cmd/host_cmd.py --timeout 660 \
  "bash utils/telegram_bot.sh recv --timeout 600"
```

Run with `block_until_ms: 0` to background it, then poll the terminal file every ~30 seconds.

3. **On reply**: read the user's instructions from the command output.

   **Echo before executing** — immediately send a short TG message paraphrasing what you understood and that you're starting work. This confirms receipt, lets the user catch misinterpretations early, and sets expectations for longer tasks. Example: "Got it: re-run the sweep with Y=5. Working on it..."

   **Multi-message handling** — if multiple messages arrived at once (newline-separated in the output), read them all before doing anything. Classify them:
   - **Refinements / corrections**: later messages update or supersede earlier ones (e.g. "use config A" then "actually use config B"). Treat only the final intent as the instruction.
   - **Distinct tasks**: messages request unrelated work (e.g. "run job A" and "check logs for job B"). Execute them sequentially.
   - **Unclear**: send a TG message listing what you received and ask which interpretation is correct. Run recv again instead of guessing.

   In all cases, the echo message should reflect your interpretation of every received message so the user sees exactly what you plan to do.

   After executing, send a new TG notification with the result. **Loop back to step 1** (send hint, run recv again).

4. **On timeout** (recv exits with code 1, output contains "Timeout"): send a final TG notification so the user knows the agent stopped listening, then end the loop. Report in the agent chat as well:

   TG message:

   > ━━━━━━━━━━━━━━━━━━━━
   > ⏱ \*Timed out — no longer listening\*
   > ⚠️ \_Back to the agent chat to continue\_
   > ━━━━━━━━━━━━━━━━━━━━

   Agent chat: "TG interactive loop ended — no reply within timeout."

5. **Early exit**: if the user's reply is just an acknowledgment ("done", "ok", "thanks", "stop"), end the loop without executing further work. Send a TG confirmation and report in the agent chat:

   TG message:

   > ━━━━━━━━━━━━━━━━━━━━
   > ✅ \*Acknowledged — no longer listening\*
   > ⚠️ \_Back to the agent chat to continue\_
   > ━━━━━━━━━━━━━━━━━━━━

   Agent chat: "TG interactive loop ended — user acknowledged."

6. **Ad-hoc timeout changes**: if the user asks to change the wait time (e.g. "increase timeout to 20 min"), adjust the `--timeout` flag on subsequent `recv` calls in the current session. Do NOT modify the script's default timeout or any docs/config — just pass `--timeout 1200` (or whatever the user requests) for the remainder of the loop.

7. **One loop at a time**: Telegram's `getUpdates` API is per-bot — any `recv` that confirms a message purges it from the queue for ALL consumers. Do NOT run interactive loops in multiple concurrent sessions with the same bot token. Only one session should use `recv` at a time; other sessions can still `send`.

8. **Progress reports during long tasks**: while executing a user's instruction (between the echo and the result message), send intermediate progress updates for tasks that take more than a few minutes. Two strategies, not mutually exclusive:

   - **Milestone-based**: report when a sub-task produces a genuinely meaningful result — something the user would care about. Routine steps ("ran a command", "read a file") are not milestones. A finding, a sub-result, or a completed phase is.
   - **Time-based**: for long tasks with no natural milestones, send periodic heartbeats with context about what you're currently doing and what you've found so far. Estimate the total duration and space heartbeats proportionally.

   **Anti-spam cap**: regardless of strategy, send no more than **2 progress messages per hour**. This forces selectivity — if many milestones occur in a short window, batch or skip most of them. Progress messages are one-way sends (no recv, no interaction hint) — do NOT re-enter the wait loop mid-task.

### Example agent flow

```
Agent: runs task, gets result
Agent: telegram_bot.sh send "*Task complete.* Result: `X`."                (Markdown)
Agent: telegram_bot.sh send "━━━━━━━━━━━━━━━━━━━━\n💬 *Awaiting further instructions*\n⏳ _Timeout: 10 minutes_\n━━━━━━━━━━━━━━━━━━━━"  (hint)
Agent: telegram_bot.sh recv --timeout 600  (backgrounded, polls terminal file)
User (on TG): "now run it again with Y=5"
Agent: reads reply from terminal output
Agent: telegram_bot.sh send "Got it: re-run with `Y`=5. Working on it..."  (echo — no hint)
Agent: executes the instruction
Agent: telegram_bot.sh send "*Done.* `Y`=5 result: `Z`."                  (Markdown)
Agent: telegram_bot.sh send "━━━━━━━━━━━━━━━━━━━━\n💬 *Awaiting further instructions*\n⏳ _Timeout: 10 minutes_\n━━━━━━━━━━━━━━━━━━━━"  (hint)
Agent: telegram_bot.sh recv --timeout 600  (backgrounded again)
... (no reply within 10 min) ...
Agent: "TG interactive loop ended — no reply within timeout."
```

## Integration with other skills

This skill is opt-in. Only use it when the user explicitly asks for notification. Typical integration points:

- **batch-sweep**: "notify me when the sweep is done" → send results table after Step 8
- **model-config**: "TG me when the test job finishes" → send job status after Step 7
- **job-triage**: "alert me if the job fails" → send failure summary

Do not proactively send notifications unless the user requested them.
