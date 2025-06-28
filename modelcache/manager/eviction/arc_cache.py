from cachetools import Cache
from collections import OrderedDict
from readerwriterlock import rwlock

_sentinel = object()

class ARC(Cache):
    """
    Adaptive Replacement Cache (ARC) implementation.

    ARC maintains four lists (T1, T2, B1, B2) to adaptively balance
    between LRU and LFU eviction strategies based on access patterns.
    """

    def __init__(self, maxsize, getsizeof=None):
        """Initialize ARC cache with maximum size."""
        super().__init__(maxsize, getsizeof)
        self.t1 = OrderedDict()  # Recent items
        self.t2 = OrderedDict()  # Frequent items
        self.b1 = OrderedDict()  # Ghost entries for T1
        self.b2 = OrderedDict()  # Ghost entries for T2
        self.p = 0               # Adaptive parameter
        self._rw_lock = rwlock.RWLockWrite()  # Thread safety

    def __len__(self):
        """Return total number of cached items."""
        return len(self.t1) + len(self.t2)

    def __contains__(self, key):
        """Check if key exists in cache."""
        return key in self.t1 or key in self.t2

    def _evict_internal(self):
        """Internal method to evict items when cache is full."""
        # Evict from cache lists to ghost lists
        while len(self.t1) + len(self.t2) > self.maxsize:
            if len(self.t1) > self.p or (len(self.t1) == 0 and len(self.t2) > 0):
                key, value = self.t1.popitem(last=False)
                self.b1[key] = value
            else:
                key, value = self.t2.popitem(last=False)
                self.b2[key] = value

        # Maintain ghost list sizes
        while len(self.b1) > (self.maxsize - self.p):
            self.b1.popitem(last=False)
        while len(self.b2) > self.p:
            self.b2.popitem(last=False)

    def __setitem__(self, key, value):
        """Insert or update a cache entry."""
        with self._rw_lock.gen_wlock():
            # Remove key from all lists first
            for l in (self.t1, self.t2, self.b1, self.b2):
                l.pop(key, None)
            # Add to recent list (T1)
            self.t1[key] = value
            self.t1.move_to_end(key)
            self._evict_internal()

    def __getitem__(self, key):
        """Retrieve a cache entry and update access pattern."""
        with self._rw_lock.gen_wlock():
            if key in self.t1:
                # Move from recent to frequent list
                value = self.t1.pop(key)
                self.t2[key] = value
                self.t2.move_to_end(key)
                self.p = max(0, self.p - 1)  # Adjust adaptive parameter
                self._evict_internal()
                return value
            if key in self.t2:
                # Access frequent list
                value = self.t2.pop(key)
                self.t2[key] = value
                self.t2.move_to_end(key)
                self.p = min(self.maxsize, self.p + 1)  # Adjust adaptive parameter
                self._evict_internal()
                return value
            if key in self.b1:
                # Promote from ghost list B1 to frequent list T2
                self.b1.pop(key)
                self.p = min(self.maxsize, self.p + 1)  # Adjust adaptive parameter
                self._evict_internal()
                value = super().__missing__(key)
                self.t2[key] = value
                self.t2.move_to_end(key)
                return value
            if key in self.b2:
                # Promote from ghost list B2 to frequent list T2
                self.b2.pop(key)
                self.p = max(0, self.p - 1)  # Adjust adaptive parameter
                self._evict_internal()
                value = super().__missing__(key)
                self.t2[key] = value
                self.t2.move_to_end(key)
                return value
            return super().__getitem__(key)

    def __missing__(self, key):
        """Handle missing keys."""
        raise KeyError(key)

    def pop(self, key, default=_sentinel):
        """Remove a cache entry."""
        with self._rw_lock.gen_wlock():
            for l in (self.t1, self.t2, self.b1, self.b2):
                if key in l:
                    return l.pop(key)
            if default is _sentinel:
                raise KeyError(key)
            return default

    def clear(self):
        """Clear all cache entries."""
        with self._rw_lock.gen_wlock():
            self.t1.clear()
            self.t2.clear()
            self.b1.clear()
            self.b2.clear()
            self.p = 0
            super().clear()

    def __iter__(self):
        """Iterate over cache keys."""
        yield from self.t1
        yield from self.t2

    def __repr__(self):
        """Return string representation of the cache."""
        return (f"ARC(maxsize={self.maxsize}, p={self.p}, len={len(self)}, "
                f"t1_len={len(self.t1)}, t2_len={len(self.t2)}, "
                f"b1_len={len(self.b1)}, b2_len={len(self.b2)})")
