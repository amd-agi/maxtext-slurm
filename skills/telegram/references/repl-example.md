# REPL Example Flow

```
Agent: runs task, gets result
Agent: telegram_bot.sh send "*Task complete.* Result: `X`."                (result)
Agent: telegram_bot.sh send "━━━...💬 *Awaiting*...⏳ _10 minutes_...━━━"  (prompt)
Agent: telegram_bot.sh recv --timeout 600  (backgrounded, polls terminal)
User (on TG): "now run it again with Y=5"
Agent: reads reply from terminal output
Agent: telegram_bot.sh send "Got it: re-run with `Y`=5. Working on it..."  (echo — no prompt)
Agent: telegram_bot.sh recv --timeout 1                                    (pre-exec peek — nothing)
Agent: starts long execution...
Agent: telegram_bot.sh send "Progress: 3/10 configs done..."               (progress)
Agent: telegram_bot.sh recv --timeout 1                                    (peek — nothing)
Agent: continues working...
Agent: telegram_bot.sh send "Progress: 6/10 configs done..."               (progress)
Agent: telegram_bot.sh recv --timeout 1                                    (peek — message found!)
User (on TG): "stop"
Agent: stops current execution
Agent: telegram_bot.sh send "*Stopped.* 6/10 configs completed: ..."       (result)
Agent: telegram_bot.sh send "━━━...💬 *Awaiting*...⏳ _10 minutes_...━━━"  (prompt)
Agent: telegram_bot.sh recv --timeout 600  (loop continues — waiting)
User (on TG): "ok run the remaining 4 with Z=3"
Agent: telegram_bot.sh send "Got it: configs 7-10 with `Z`=3..."           (echo)
Agent: executes...
Agent: telegram_bot.sh send "*Done.* Remaining 4 configs complete: ..."    (result)
Agent: telegram_bot.sh send "━━━...💬 *Awaiting*...⏳ _10 minutes_...━━━"  (prompt)
Agent: telegram_bot.sh recv --timeout 600  (waiting again)
User (on TG): "thanks"
Agent: telegram_bot.sh send "━━━...💬 *Awaiting*...⏳ _10 minutes_...━━━"  (ack → prompt only, loop back)
Agent: telegram_bot.sh recv --timeout 600  (still listening)
User (on TG): "stop listening"
Agent: telegram_bot.sh send "━━━...✅ *Acknowledged*...━━━"                (explicit exit)
Agent: "TG interactive loop ended — user requested."
```
