#!/bin/bash
#
# Fix IB partition key for fabrics where default P_Key is limited-member.
#
# Problem: If pkey[0] = 0x7fff (limited member), two endpoints both using
#          pkey_index=0 cannot communicate (IB spec requires at least one
#          full member). This causes IBV_WC_RETRY_EXC_ERR(12).
#
# Solution: Build a tiny LD_PRELOAD shim that intercepts ibv_modify_qp
#           and replaces pkey_index 0 with the first full-member index.
#
# This is a no-op when the default P_Key is already a full member (0xffff).

_IB_PKEY_FIX_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

_ib_pkey_needs_fix() {
    local pkey_dir="/sys/class/infiniband"
    [[ -d "$pkey_dir" ]] || return 1

    local first_hca
    first_hca=$(ls "$pkey_dir" 2>/dev/null | head -1)
    [[ -n "$first_hca" ]] || return 1

    local default_pkey
    default_pkey=$(cat "$pkey_dir/$first_hca/ports/1/pkeys/0" 2>/dev/null)
    [[ -n "$default_pkey" ]] || return 1

    local val=$(( default_pkey ))
    if (( val & 0x8000 )); then
        return 1  # already full member
    fi
    return 0  # needs fix
}

_ib_pkey_find_full_member_index() {
    local pkey_dir="/sys/class/infiniband"
    local first_hca
    first_hca=$(ls "$pkey_dir" 2>/dev/null | head -1)

    local idx
    for f in "$pkey_dir/$first_hca/ports/1/pkeys"/*; do
        idx=$(basename "$f")
        [[ "$idx" == "0" ]] && continue
        local val
        val=$(cat "$f" 2>/dev/null)
        [[ -n "$val" && "$val" != "0x0000" ]] || continue
        local int_val=$(( val ))
        if (( int_val & 0x8000 )); then
            echo "$idx"
            return 0
        fi
    done
    return 1
}

_ib_pkey_build_shim() {
    local target_idx="$1"
    local shim_dir="${_IB_PKEY_FIX_DIR}/.ib_pkey_shim"
    local shim_so="$shim_dir/ib_pkey_fix.so"

    if [[ -f "$shim_so" ]]; then
        echo "$shim_so"
        return 0
    fi

    mkdir -p "$shim_dir"

    cat > "$shim_dir/ib_pkey_fix.c" << 'SHIM_EOF'
#define _GNU_SOURCE
#include <dlfcn.h>
#include <infiniband/verbs.h>
#include <stdlib.h>
#include <stdio.h>

static int target_pkey_index = -1;
static int (*real_fn)(struct ibv_qp *, struct ibv_qp_attr *, int) = NULL;

static int (*resolve_real_fn(void))(struct ibv_qp *, struct ibv_qp_attr *, int) {
    typedef int (*fn_t)(struct ibv_qp *, struct ibv_qp_attr *, int);
    fn_t f;

    f = (fn_t)dlsym(RTLD_NEXT, "ibv_modify_qp");
    if (f && f != (fn_t)ibv_modify_qp) return f;

    static const char *libs[] = {
        "libibverbs.so.1", "libibverbs.so", NULL
    };
    for (const char **p = libs; *p; p++) {
        void *h = dlopen(*p, RTLD_NOW | RTLD_NOLOAD);
        if (!h) h = dlopen(*p, RTLD_NOW);
        if (h) {
            f = (fn_t)dlsym(h, "ibv_modify_qp");
            if (f && f != (fn_t)ibv_modify_qp) return f;
        }
    }
    return NULL;
}

static void __attribute__((constructor)) init(void) {
    const char *env = getenv("_IB_PKEY_FIX_INDEX");
    if (env) target_pkey_index = atoi(env);
    real_fn = resolve_real_fn();
    if (target_pkey_index >= 0 && real_fn)
        fprintf(stderr, "[ib_pkey_fix] shim active: pkey_index 0 -> %d\n",
                target_pkey_index);
    else if (target_pkey_index >= 0)
        fprintf(stderr, "[ib_pkey_fix] WARNING: could not resolve real "
                "ibv_modify_qp — shim disabled\n");
}

int ibv_modify_qp(struct ibv_qp *qp, struct ibv_qp_attr *attr, int attr_mask) {
    if (!real_fn) {
        real_fn = resolve_real_fn();
        if (!real_fn) return EINVAL;
    }
    if (target_pkey_index >= 0 &&
        (attr_mask & IBV_QP_PKEY_INDEX) &&
        attr->pkey_index == 0) {
        attr->pkey_index = target_pkey_index;
    }
    return real_fn(qp, attr, attr_mask);
}
SHIM_EOF

    if gcc -shared -fPIC -o "$shim_so" "$shim_dir/ib_pkey_fix.c" -ldl -I/usr/include 2>/dev/null; then
        echo "$shim_so"
        return 0
    fi
    return 1
}

if _ib_pkey_needs_fix; then
    _full_idx=$(_ib_pkey_find_full_member_index)
    if [[ -n "$_full_idx" ]]; then
        _pkey_val=$(cat "/sys/class/infiniband/$(ls /sys/class/infiniband | head -1)/ports/1/pkeys/$_full_idx" 2>/dev/null)
        _shim_so=$(_ib_pkey_build_shim "$_full_idx")
        if [[ -n "$_shim_so" && -f "$_shim_so" ]]; then
            export _IB_PKEY_FIX_INDEX="$_full_idx"
            export MAXTEXT_LD_PRELOAD="${MAXTEXT_LD_PRELOAD:+$MAXTEXT_LD_PRELOAD:}$_shim_so"
            echo "[IB P_Key fix] Default P_Key is limited-member; patching pkey_index 0 -> $_full_idx (P_Key=$_pkey_val) via LD_PRELOAD"
        else
            echo "[IB P_Key fix] WARNING: Failed to build shim; IB may fail with IBV_WC_RETRY_EXC_ERR(12)" >&2
        fi
        unset _pkey_val _shim_so
    else
        echo "[IB P_Key fix] WARNING: No full-member P_Key found in table; IB may fail" >&2
    fi
    unset _full_idx
else
    echo "[IB P_Key fix] Default P_Key is full-member; no fix needed"
fi

unset _IB_PKEY_FIX_DIR
