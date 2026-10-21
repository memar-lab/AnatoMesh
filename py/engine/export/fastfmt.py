"""Fast byte-exact deck-row formatting for AnatoMesh (numba kernels).

Each float is formatted into a fixed-width byte slot, reproducing C printf
``%.6g`` / ``%.6f`` by scaling to an integer of 6 significant digits (or
6 decimals) and rounding it with ``rint``. Any value whose rounding cannot be
proven identical to the exact binary-value rounding of printf (near ties, near
decade boundaries, huge/tiny/non-finite values) is flagged. Flagged values are
re-formatted with Python's ``%`` in a second pass and spliced into the slots.
Integers are written straight into the output buffer.

Assembly kernels compute per-row byte lengths, take an exclusive prefix sum to
get row offsets, allocate one uint8 buffer, and fill rows in parallel. Rows are
separated by a single b"\\n" with no trailing newline, replacing "\\n".join(rows).
"""
import math

import numpy as np
from numba import njit, prange

_G6_WIDTH = 16
_F6_WIDTH = 24

# Correctly rounded powers of ten (index = exponent), 1e0 .. 1e308.
_P10 = np.array([float("1e%d" % i) for i in range(309)], dtype=np.float64)
_I10 = np.array([10 ** i for i in range(7)], dtype=np.int64)


# ---------------------------------------------------------------- integers

@njit(cache=True)
def _abs_u(v):
    if v < 0:
        return np.uint64(-(v + 1)) + np.uint64(1)  # safe for int64 min
    return np.uint64(v)


@njit(cache=True)
def _ulen(u):
    ten = np.uint64(10)
    n = 1
    while u >= ten:
        u //= ten
        n += 1
    return n


@njit(cache=True)
def _ilen(v):
    n = _ulen(_abs_u(v))
    if v < 0:
        n += 1
    return n


@njit(cache=True)
def _put_u(buf, pos, u, n):
    ten = np.uint64(10)
    j = pos + n - 1
    while j >= pos:
        buf[j] = np.uint8(48) + np.uint8(u % ten)
        u //= ten
        j -= 1


@njit(cache=True)
def _put_i(buf, pos, v):
    if v < 0:
        buf[pos] = 45
        pos += 1
    u = _abs_u(v)
    n = _ulen(u)
    _put_u(buf, pos, u, n)
    return pos + n


@njit(cache=True)
def _copy(buf, pos, slot, n):
    for j in range(n):
        buf[pos + j] = slot[j]
    return pos + n


# ------------------------------------------------------------------- %.6g

@njit(cache=True)
def _scale(ax, k):
    # Multiply by 10**-k for negative k: avoids dividing by a tiny number.
    if k >= 0:
        return ax / _P10[k]
    return ax * _P10[-k]


@njit(cache=True)
def _g6_one(x, out):
    """Format x like '%.6g' into out. Returns (length, flagged)."""
    if x == 0.0:
        pos = 0
        if math.copysign(1.0, x) < 0.0:
            out[0] = 45
            pos = 1
        out[pos] = 48
        return pos + 1, False
    ax = abs(x)
    if not (ax >= 1e-300 and ax <= 1e300):  # also catches nan / inf
        return 0, True
    lg = math.log10(ax)
    if abs(lg - math.floor(lg + 0.5)) < 1e-9:  # too close to a decade boundary
        return 0, True
    e0 = int(math.floor(lg))
    adj = 0
    m = 0.0
    while True:
        s = _scale(ax, e0 - 5)
        fl = math.floor(s)
        if (abs(s - fl - 0.5) < 1e-6 or abs(s - 100000.0) < 1e-6
                or abs(s - 999999.5) < 1e-6):
            return 0, True
        m = np.rint(s)
        if m >= 1000000.0:
            e0 += 1
        elif m < 100000.0:
            e0 -= 1
        else:
            break
        adj += 1
        if adj >= 2:
            return 0, True

    mi = np.int64(m)  # six digits, 100000 .. 999999
    pos = 0
    if x < 0.0:
        out[0] = 45
        pos = 1
    nd = 5  # index of last nonzero digit
    while nd > 0 and (mi // _I10[5 - nd]) % 10 == 0:
        nd -= 1

    if e0 < -4 or e0 >= 6:
        out[pos] = 48 + mi // 100000
        pos += 1
        if nd >= 1:
            out[pos] = 46
            pos += 1
            for j in range(1, nd + 1):
                out[pos] = 48 + (mi // _I10[5 - j]) % 10
                pos += 1
        out[pos] = 101
        pos += 1
        if e0 < 0:
            out[pos] = 45
            ae = -e0
        else:
            out[pos] = 43
            ae = e0
        pos += 1
        if ae >= 100:
            out[pos] = 48 + ae // 100
            pos += 1
            ae = ae % 100
        out[pos] = 48 + ae // 10
        out[pos + 1] = 48 + ae % 10
        pos += 2
    elif e0 >= 0:
        for j in range(0, e0 + 1):
            out[pos] = 48 + (mi // _I10[5 - j]) % 10
            pos += 1
        if nd > e0:
            out[pos] = 46
            pos += 1
            for j in range(e0 + 1, nd + 1):
                out[pos] = 48 + (mi // _I10[5 - j]) % 10
                pos += 1
    else:
        out[pos] = 48
        out[pos + 1] = 46
        pos += 2
        for _z in range(-e0 - 1):
            out[pos] = 48
            pos += 1
        for j in range(0, nd + 1):
            out[pos] = 48 + (mi // _I10[5 - j]) % 10
            pos += 1
    return pos, False


@njit(parallel=True, cache=True)
def _g6_slots(x):
    n = x.shape[0]
    slots = np.zeros((n, _G6_WIDTH), np.uint8)
    lens = np.zeros(n, np.int32)
    flags = np.zeros(n, np.bool_)
    for ii in prange(n):
        i = np.int64(ii)
        ln, fl = _g6_one(x[i], slots[i])
        lens[i] = ln
        flags[i] = fl
    return slots, lens, flags


# ------------------------------------------------------------------- %.Nf

@njit(cache=True)
def _fn_one(x, out, nd):
    """Format x like '%.<nd>f' (nd 1..6) into out. Returns (length, flagged)."""
    v = abs(x)
    # Limit to v < 1e5 so that t = v * 10**nd < 1e11 carries an absolute error
    # below ~1.1e-5, far inside the 1e-4 tie margin below. Also catches nan/inf.
    if not (v < 1e5):
        return 0, True
    scale = _I10[nd]
    t = v * _P10[nd]
    if abs((t - math.floor(t)) - 0.5) < 1e-4:
        return 0, True
    r = np.int64(np.rint(t))
    ip = r // scale
    fr = r % scale
    pos = 0
    if math.copysign(1.0, x) < 0.0:  # "-0.000000" for tiny negatives and -0.0
        out[0] = 45
        pos = 1
    u = np.uint64(ip)
    n = _ulen(u)
    _put_u(out, pos, u, n)
    pos += n
    out[pos] = 46
    pos += 1
    for j in range(nd - 1, -1, -1):
        out[pos + j] = 48 + fr % 10
        fr //= 10
    return pos + nd, False


@njit(parallel=True, cache=True)
def _fn_slots(x, nd):
    n = x.shape[0]
    slots = np.zeros((n, _F6_WIDTH), np.uint8)
    lens = np.zeros(n, np.int32)
    flags = np.zeros(n, np.bool_)
    for ii in prange(n):
        i = np.int64(ii)
        ln, fl = _fn_one(x[i], slots[i], nd)
        lens[i] = ln
        flags[i] = fl
    return slots, lens, flags


# --------------------------------------------------------- Python fallback

def _fix_flags(x, slots, lens, flags, fmt):
    """Re-format flagged entries with Python's % and splice them into slots.

    Slots are widened if a fallback string does not fit (e.g. '%.6f' % 1e100).
    Returns (slots, lens).
    """
    idx = np.flatnonzero(flags)
    if idx.size == 0:
        return slots, lens
    strs = [(fmt % float(x[j])).encode("ascii") for j in idx]
    need = max(len(s) for s in strs)
    if need > slots.shape[1]:
        wide = np.zeros((slots.shape[0], need), np.uint8)
        wide[:, :slots.shape[1]] = slots
        slots = wide
    for j, s in zip(idx, strs):
        slots[j, :len(s)] = np.frombuffer(s, np.uint8)
        lens[j] = len(s)
    return slots, lens


def _float_slots(flat, kind):
    if kind == "g":
        slots, lens, flags = _g6_slots(flat)
        fmt = "%.6g"
    else:
        slots, lens, flags = _fn_slots(flat, 6)
        fmt = "%.6f"
    return _fix_flags(flat, slots, lens, flags, fmt)


def _format_list(values, kind, nd=6):
    arr = np.ascontiguousarray(values, dtype=np.float64).ravel()
    n = arr.shape[0]
    if n == 0:
        return []
    if kind == "g":
        slots, lens, flags = _g6_slots(arr)
        fmt = "%.6g"
    else:
        if not 1 <= nd <= 6:
            raise ValueError("fixed() takes 1 to 6 decimals")
        slots, lens, flags = _fn_slots(arr, nd)
        fmt = "%%.%df" % nd
    raw = slots.tobytes()
    w = slots.shape[1]
    out = [raw[i * w:i * w + ln].decode("ascii")
           for i, ln in enumerate(lens.tolist())]
    for j in np.flatnonzero(flags).tolist():
        out[j] = fmt % float(arr[j])
    return out


def g6(values):
    """Format values like '%.6g' % v; returns list[str]."""
    return _format_list(values, "g")


def f6(values):
    """Format values like '%.6f' % v; returns list[str]."""
    return _format_list(values, "f")


def fixed(values, decimals):
    """Format values like '%.<decimals>f' % v (1 to 6 decimals); returns list[str]."""
    return _format_list(values, "f", decimals)


# ------------------------------------------------------- assembly kernels

@njit(parallel=True, cache=True)
def _asm_nodes(start, sx, lx):
    n = lx.shape[0] // 3
    rl = np.empty(n, np.int64)
    for ii in prange(n):
        i = np.int64(ii)
        # nid , x , y , z ,0,0  -> 3 commas + 4 fixed chars
        rl[i] = _ilen(start + i) + 7 + lx[3 * i] + lx[3 * i + 1] + lx[3 * i + 2]
    off = np.empty(n + 1, np.int64)
    off[0] = 0
    for i in range(n):
        off[i + 1] = off[i] + rl[i] + 1
    buf = np.empty(off[n] - 1, np.uint8)
    for ii in prange(n):
        i = np.int64(ii)
        p = off[i]
        p = _put_i(buf, p, start + i)
        for c in range(3):
            buf[p] = 44
            p += 1
            p = _copy(buf, p, sx[3 * i + c], lx[3 * i + c])
        buf[p] = 44
        buf[p + 1] = 48
        buf[p + 2] = 44
        buf[p + 3] = 48
        p += 4
        if i < n - 1:
            buf[p] = 10
    return buf


@njit(parallel=True, cache=True)
def _asm_int(cells, start, has_pid, pid):
    n = cells.shape[0]
    k = cells.shape[1]
    pl = 0
    if has_pid:
        pl = _ilen(pid) + 1
    rl = np.empty(n, np.int64)
    for ii in prange(n):
        i = np.int64(ii)
        tot = _ilen(start + i) + pl
        for j in range(k):
            tot += 1 + _ilen(cells[i, j])
        rl[i] = tot
    off = np.empty(n + 1, np.int64)
    off[0] = 0
    for i in range(n):
        off[i + 1] = off[i] + rl[i] + 1
    buf = np.empty(off[n] - 1, np.uint8)
    for ii in prange(n):
        i = np.int64(ii)
        p = off[i]
        p = _put_i(buf, p, start + i)
        if has_pid:
            buf[p] = 44
            p += 1
            p = _put_i(buf, p, pid)
        for j in range(k):
            buf[p] = 44
            p += 1
            p = _put_i(buf, p, cells[i, j])
        if i < n - 1:
            buf[p] = 10
    return buf


@njit(parallel=True, cache=True)
def _asm_ortho(cells, start, pid, sa, la, sd, ld):
    n = cells.shape[0]
    m = cells.shape[1]
    plen = _ilen(pid)
    rl = np.empty(n, np.int64)
    for ii in prange(n):
        i = np.int64(ii)
        nl = 0
        for j in range(m):
            nl += _ilen(cells[i, j])
        if m > 1:
            nl += m - 1
        al = la[3 * i] + la[3 * i + 1] + la[3 * i + 2] + 2
        dl = ld[3 * i] + ld[3 * i + 1] + ld[3 * i + 2] + 2
        # "eid,pid" + nl + nl_a + nl_d + 3 inner newlines
        rl[i] = _ilen(start + i) + 1 + plen + nl + al + dl + 3
    off = np.empty(n + 1, np.int64)
    off[0] = 0
    for i in range(n):
        off[i + 1] = off[i] + rl[i] + 1
    buf = np.empty(off[n] - 1, np.uint8)
    for ii in prange(n):
        i = np.int64(ii)
        p = off[i]
        p = _put_i(buf, p, start + i)
        buf[p] = 44
        p += 1
        p = _put_i(buf, p, pid)
        buf[p] = 10
        p += 1
        for j in range(m):
            if j > 0:
                buf[p] = 44
                p += 1
            p = _put_i(buf, p, cells[i, j])
        buf[p] = 10
        p += 1
        for c in range(3):
            if c > 0:
                buf[p] = 44
                p += 1
            p = _copy(buf, p, sa[3 * i + c], la[3 * i + c])
        buf[p] = 10
        p += 1
        for c in range(3):
            if c > 0:
                buf[p] = 44
                p += 1
            p = _copy(buf, p, sd[3 * i + c], ld[3 * i + c])
        if i < n - 1:
            buf[p] = 10
    return buf


# --------------------------------------------------------------- public API

def node_rows(points, start):
    """b"\\n".join("%d,%.6g,%.6g,%.6g,0,0" % (nid, x, y, z)), nid from start."""
    pts = np.ascontiguousarray(points, dtype=np.float64)
    if pts.shape[0] == 0:
        return b""
    flat = pts.reshape(-1)
    slots, lens = _float_slots(flat, "g")
    return _asm_nodes(np.int64(start), slots, lens).tobytes()


def int_rows(cells, start, pid=None):
    """Integer rows: eid, [pid,] c0, c1, ... joined by commas, rows by newline."""
    c = np.ascontiguousarray(cells, dtype=np.int64)
    if c.shape[0] == 0:
        return b""
    if c.ndim == 1:
        c = c.reshape(c.shape[0], 1)
    has_pid = pid is not None
    return _asm_int(c, np.int64(start), has_pid,
                    np.int64(pid if has_pid else 0)).tobytes()


def ortho_rows(cells, start, pid, a, d):
    """Four lines per element: 'eid,pid', node line, '%.6f' a-vector, d-vector."""
    c = np.ascontiguousarray(cells, dtype=np.int64)
    if c.shape[0] == 0:
        return b""
    if c.ndim == 1:
        c = c.reshape(c.shape[0], 1)
    av = np.ascontiguousarray(a, dtype=np.float64).reshape(-1)
    dv = np.ascontiguousarray(d, dtype=np.float64).reshape(-1)
    sa, la = _float_slots(av, "f")
    sd, ld = _float_slots(dv, "f")
    return _asm_ortho(c, np.int64(start), np.int64(pid), sa, la, sd, ld).tobytes()


@njit(parallel=True, cache=True)
def _asm_fa(start, name, sf, lf, st, lt):
    n = lf.shape[0]
    rl = np.empty(n, np.int64)
    for ii in prange(n):
        i = np.int64(ii)
        rl[i] = (_ilen(start + i) + 1 + name.shape[0] + 1 + lf[i]
                 + 3 + lt[3 * i] + lt[3 * i + 1] + lt[3 * i + 2])
    off = np.empty(n + 1, np.int64)
    off[0] = 0
    for i in range(n):
        off[i + 1] = off[i] + rl[i] + 1
    buf = np.empty(off[n] - 1, np.uint8)
    for ii in prange(n):
        i = np.int64(ii)
        p = _put_i(buf, off[i], start + i)
        buf[p] = 44
        p = _copy(buf, p + 1, name, name.shape[0])
        buf[p] = 44
        p = _copy(buf, p + 1, sf[i], lf[i])
        for c in range(3):
            buf[p] = 44
            p = _copy(buf, p + 1, st[3 * i + c], lt[3 * i + c])
        if i < n - 1:
            buf[p] = 10
    return buf


def fa_rows(start, name, fa, tangent):
    """Rows 'eid,name,%.4f fa,%.4f x,y,z' of the axon FA sidecar; a non-finite fa is empty."""
    fa = np.ascontiguousarray(fa, dtype=np.float64)
    if fa.shape[0] == 0:
        return b""
    bad = ~np.isfinite(fa)
    sf, lf, ff = _fn_slots(np.where(bad, 0.0, fa), 4)
    sf, lf = _fix_flags(fa, sf, lf, ff & ~bad, "%.4f")
    lf[bad] = 0
    tv = np.ascontiguousarray(tangent, dtype=np.float64).reshape(-1)
    st, lt, ft = _fn_slots(tv, 4)
    st, lt = _fix_flags(tv, st, lt, ft, "%.4f")
    label = np.frombuffer(name.encode("ascii"), np.uint8).copy()
    return _asm_fa(np.int64(start), label, sf, lf, st, lt).tobytes()
