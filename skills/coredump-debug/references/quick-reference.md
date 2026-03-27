# Quick Reference: GDB Commands for Coredumps

| Command | Purpose |
|---|---|
| `bt full` | Full backtrace with local variables |
| `frame N` | Switch to frame N |
| `info locals` | Show local variables in current frame |
| `info registers` | Show CPU registers |
| `info threads` | List all threads |
| `thread N` | Switch to thread N |
| `thread apply all bt` | Backtrace for every thread |
| `x/10i $rip` | Disassemble 10 instructions at crash point |
| `x/s ADDR` | Print string at address |
| `p expr` | Evaluate expression |
| `info proc mappings` | Show library load addresses |

# Quick Reference: Crash Exit Codes

| Exit Code | Signal | Meaning |
|---|---|---|
| 139 | SIGSEGV (11) | Segmentation fault (invalid memory access) |
| 134 | SIGABRT (6) | Abort (assertion failure, double free) |
| 136 | SIGFPE (8) | Floating point exception (division by zero) |
| 137 | SIGKILL (9) | Killed (OOM killer, timeout) |
| 138 | SIGBUS (7) | Bus error (misaligned access, bad mmap) |
