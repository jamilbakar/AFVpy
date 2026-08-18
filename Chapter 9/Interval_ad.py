import math

class Interval:
    __array_priority__ = 1000
    def __init__(self, lo, hi=None):
        if hi is None: hi = lo
        self.lo = float(min(lo, hi)); self.hi = float(max(lo, hi))
    def __repr__(self): return f"[{self.lo}, {self.hi}]"
    def __add__(s, o): o = _iv(o); return Interval(s.lo + o.lo, s.hi + o.hi)
    __radd__ = __add__
    def __sub__(s, o): o = _iv(o); return Interval(s.lo - o.hi, s.hi - o.lo)
    def __rsub__(s, o): return _iv(o).__sub__(s)
    def __neg__(s): return Interval(-s.hi, -s.lo)
    def __mul__(s, o):
        o = _iv(o); p = [s.lo*o.lo, s.lo*o.hi, s.hi*o.lo, s.hi*o.hi]
        return Interval(min(p), max(p))
    __rmul__ = __mul__
    def __truediv__(s, o):
        o = _iv(o); p = [s.lo/o.lo, s.lo/o.hi, s.hi/o.lo, s.hi/o.hi]
        return Interval(min(p), max(p))
    def __pow__(s, n):
        assert isinstance(n, int) and n >= 0
        r = Interval(1.0)
        for _ in range(n): r = r * s
        return r

def _iv(x): return x if isinstance(x, Interval) else Interval(x)

def sin(x):
    x = _iv(x)
    lo, hi = x.lo, x.hi
    vals = [math.sin(lo), math.sin(hi)]
    # include critical points pi/2 + k*pi in [lo,hi]
    k = math.floor((lo - math.pi/2)/math.pi)
    while math.pi/2 + k*math.pi <= hi + 1e-12:
        c = math.pi/2 + k*math.pi
        if lo - 1e-12 <= c <= hi + 1e-12: vals.append(math.sin(c))
        k += 1
    return Interval(min(vals), max(vals))

def cos(x): return sin(_iv(x) + math.pi/2)

def mid(i): return (i.lo + i.hi) / 2.0

# forward-mode AD over intervals: Dual carries value + gradient (both intervals)
class Dual:
    __array_priority__ = 1000
    def __init__(self, val, grad):
        self.val = _iv(val); self.grad = [_iv(g) for g in grad]
    def __add__(s, o):
        o = _dual(o, len(s.grad)); return Dual(s.val+o.val, [a+b for a,b in zip(s.grad,o.grad)])
    __radd__ = __add__
    def __sub__(s, o):
        o = _dual(o, len(s.grad)); return Dual(s.val-o.val, [a-b for a,b in zip(s.grad,o.grad)])
    def __rsub__(s, o): return _dual(o, len(s.grad)).__sub__(s)
    def __neg__(s): return Dual(-s.val, [-g for g in s.grad])
    def __mul__(s, o):
        o = _dual(o, len(s.grad))
        return Dual(s.val*o.val, [s.val*b + a*o.val for a,b in zip(s.grad,o.grad)])
    __rmul__ = __mul__
    def __truediv__(s, o):
        o = _dual(o, len(s.grad))
        return Dual(s.val/o.val, [(a*o.val - s.val*b)/(o.val*o.val) for a,b in zip(s.grad,o.grad)])
    def __pow__(s, n):
        r = Dual(Interval(1.0), [Interval(0.0)]*len(s.grad))
        for _ in range(n): r = r * s
        return r

def _dual(x, n):
    if isinstance(x, Dual): return x
    return Dual(_iv(x), [Interval(0.0)]*n)

def dsin(d):
    if not isinstance(d, Dual): return sin(d)
    c = cos(d.val); return Dual(sin(d.val), [c*g for g in d.grad])
def dcos(d):
    if not isinstance(d, Dual): return cos(d)
    ns = -sin(d.val); return Dual(cos(d.val), [ns*g for g in d.grad])

def gradient(f, x):
    # x: list of Intervals. Returns list of Intervals (grad of scalar f).
    n = len(x)
    seeds = [Dual(x[i], [Interval(1.0 if j==i else 0.0) for j in range(n)]) for i in range(n)]
    out = f(seeds)
    return out.grad

def jacobian(f, x):
    n = len(x)
    seeds = [Dual(x[i], [Interval(1.0 if j==i else 0.0) for j in range(n)]) for i in range(n)]
    outs = f(seeds)
    return [o.grad for o in outs]


# second-order forward AD over intervals: value, gradient, Hessian (all intervals)
class HDual:
    __array_priority__ = 1000
    def __init__(self, val, g, H):
        self.val = _iv(val); self.g = [_iv(x) for x in g]
        self.H = [[_iv(x) for x in row] for row in H]
    def _n(self): return len(self.g)
    def __add__(s, o):
        o = _hd(o, s._n())
        return HDual(s.val+o.val, [a+b for a,b in zip(s.g,o.g)],
                     [[s.H[i][j]+o.H[i][j] for j in range(s._n())] for i in range(s._n())])
    __radd__ = __add__
    def __sub__(s, o):
        o = _hd(o, s._n())
        return HDual(s.val-o.val, [a-b for a,b in zip(s.g,o.g)],
                     [[s.H[i][j]-o.H[i][j] for j in range(s._n())] for i in range(s._n())])
    def __rsub__(s, o): return _hd(o, s._n()).__sub__(s)
    def __neg__(s):
        n=s._n(); return HDual(-s.val, [-x for x in s.g], [[-s.H[i][j] for j in range(n)] for i in range(n)])
    def __mul__(s, o):
        o = _hd(o, s._n()); n = s._n()
        g = [s.val*o.g[i] + s.g[i]*o.val for i in range(n)]
        H = [[s.val*o.H[i][j] + s.g[i]*o.g[j] + s.g[j]*o.g[i] + s.H[i][j]*o.val
              for j in range(n)] for i in range(n)]
        return HDual(s.val*o.val, g, H)
    __rmul__ = __mul__
    def __pow__(s, k):
        n = s._n()
        r = HDual(Interval(1.0), [Interval(0.0)]*n, [[Interval(0.0)]*n for _ in range(n)])
        for _ in range(k): r = r * s
        return r

def _hd(x, n):
    if isinstance(x, HDual): return x
    return HDual(_iv(x), [Interval(0.0)]*n, [[Interval(0.0)]*n for _ in range(n)])

def hdsin(d):
    if not isinstance(d, HDual): return sin(d)
    n = d._n(); f = sin(d.val); fp = cos(d.val); fpp = -sin(d.val)
    g = [fp*d.g[i] for i in range(n)]
    H = [[fpp*d.g[i]*d.g[j] + fp*d.H[i][j] for j in range(n)] for i in range(n)]
    return HDual(f, g, H)
def hdcos(d):
    if not isinstance(d, HDual): return cos(d)
    n = d._n(); f = cos(d.val); fp = -sin(d.val); fpp = -cos(d.val)
    g = [fp*d.g[i] for i in range(n)]
    H = [[fpp*d.g[i]*d.g[j] + fp*d.H[i][j] for j in range(n)] for i in range(n)]
    return HDual(f, g, H)

def hessian(f, x):
    n = len(x)
    seeds = [HDual(x[i], [Interval(1.0 if j==i else 0.0) for j in range(n)],
                   [[Interval(0.0)]*n for _ in range(n)]) for i in range(n)]
    out = f(seeds)
    return out.H


def psin(x):  # polymorphic sin: dispatches on Interval / Dual / HDual / number
    if isinstance(x, HDual): return hdsin(x)
    if isinstance(x, Dual): return dsin(x)
    return sin(x)

def pcos(x):
    if isinstance(x, HDual): return hdcos(x)
    if isinstance(x, Dual): return dcos(x)
    return cos(x)