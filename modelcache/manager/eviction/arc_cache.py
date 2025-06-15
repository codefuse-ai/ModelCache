from cachetools import Cache
from collections import OrderedDict
from readerwriterlock import rwlock

_sentinel = object()

class ARC(Cache):
    def __init__(self, maxsize, getsizeof=None):
        super().__init__(maxsize, getsizeof)
        self.t1 = OrderedDict()
        self.t2 = OrderedDict()
        self.b1 = OrderedDict()
        self.b2 = OrderedDict()
        self.p = 0
        self._rw_lock = rwlock.RWLockWrite()

    def __len__(self):
        return len(self.t1) + len(self.t2)

    def __contains__(self, key):
        return key in self.t1 or key in self.t2

    def _evict_internal(self):
        while len(self.t1) + len(self.t2) > self.maxsize:
            if len(self.t1) > self.p or (len(self.t1) == 0 and len(self.t2) > 0):
                key, value = self.t1.popitem(last=False)
                self.b1[key] = value
            else:
                key, value = self.t2.popitem(last=False)
                self.b2[key] = value
        while len(self.b1) > (self.maxsize - self.p):
            self.b1.popitem(last=False)
        while len(self.b2) > self.p:
            self.b2.popitem(last=False)

    def __setitem__(self, key, value):
        with self._rw_lock.gen_wlock():
            for l in (self.t1, self.t2, self.b1, self.b2):
                l.pop(key, None)
            self.t1[key] = value
            self.t1.move_to_end(key)
            self._evict_internal()

    def __getitem__(self, key):
        with self._rw_lock.gen_wlock():
            if key in self.t1:
                value = self.t1.pop(key)
                self.t2[key] = value
                self.t2.move_to_end(key)
                self.p = max(0, self.p - 1)
                self._evict_internal()
                return value
            if key in self.t2:
                value = self.t2.pop(key)
                self.t2[key] = value
                self.t2.move_to_end(key)
                self.p = min(self.maxsize, self.p + 1)
                self._evict_internal()
                return value
            if key in self.b1:
                self.b1.pop(key)
                self.p = min(self.maxsize, self.p + 1)
                self._evict_internal()
                value = super().__missing__(key)
                self.t2[key] = value
                self.t2.move_to_end(key)
                return value
            if key in self.b2:
                self.b2.pop(key)
                self.p = max(0, self.p - 1)
                self._evict_internal()
                value = super().__missing__(key)
                self.t2[key] = value
                self.t2.move_to_end(key)
                return value
            return super().__getitem__(key)

    def __missing__(self, key):
        raise KeyError(key)

    def pop(self, key, default=_sentinel):
        with self._rw_lock.gen_wlock():
            for l in (self.t1, self.t2, self.b1, self.b2):
                if key in l:
                    return l.pop(key)
            if default is _sentinel:
                raise KeyError(key)
            return default

    def clear(self):
        with self._rw_lock.gen_wlock():
            self.t1.clear()
            self.t2.clear()
            self.b1.clear()
            self.b2.clear()
            self.p = 0
            super().clear()

    def __iter__(self):
        yield from self.t1
        yield from self.t2

    def __repr__(self):
        return (f"ARC(maxsize={self.maxsize}, p={self.p}, len={len(self)}, "
                f"t1_len={len(self.t1)}, t2_len={len(self.t2)}, "
                f"b1_len={len(self.b1)}, b2_len={len(self.b2)})")
