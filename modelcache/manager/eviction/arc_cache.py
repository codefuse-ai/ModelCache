from cachetools import Cache
from collections import OrderedDict

class ARC(Cache):
    """
    Adaptive Replacement Cache (ARC) implementation with on_evict callback.
    Balances recency and frequency via two active lists (T1, T2) and two ghost lists (B1, B2).
    Calls on_evict([key]) whenever an item is evicted from the active cache.
    """

    def __init__(self, maxsize, getsizeof=None, on_evict=None):
        """
        Args:
            maxsize (int): Maximum cache size.
            getsizeof (callable, optional): Sizing function for items.
            on_evict (callable, optional): Callback called as on_evict([key]) when a key is evicted.
        """
        super().__init__(maxsize, getsizeof)
        self.t1 = OrderedDict()
        self.t2 = OrderedDict()
        self.b1 = OrderedDict()
        self.b2 = OrderedDict()
        self.p = 0  # Adaptive target for T1 size.
        self.on_evict = on_evict

    def __len__(self):
        return len(self.t1) + len(self.t2)

    def __contains__(self, key):
        return key in self.t1 or key in self.t2

    def _evict_internal(self):
        """
        Evicts items from T1 or T2 if cache is over capacity, and prunes ghost lists.
        Calls on_evict for each evicted key.
        """
        # Evict from T1 or T2 if active cache > maxsize
        while len(self.t1) + len(self.t2) > self.maxsize:
            if len(self.t1) > self.p or (len(self.t1) == 0 and len(self.t2) > 0):
                key, value = self.t1.popitem(last=False)
                self.b1[key] = value
                if self.on_evict:
                    self.on_evict([key])
            else:
                key, value = self.t2.popitem(last=False)
                self.b2[key] = value
                if self.on_evict:
                    self.on_evict([key])
        # Prune ghost lists to their max lengths
        while len(self.b1) > (self.maxsize - self.p):
            self.b1.popitem(last=False)
        while len(self.b2) > self.p:
            self.b2.popitem(last=False)

    def __setitem__(self, key, value):
        # Remove from all lists before re-inserting
        for l in (self.t1, self.t2, self.b1, self.b2):
            l.pop(key, None)
        self.t1[key] = value
        self.t1.move_to_end(key)
        self._evict_internal()

    def __getitem__(self, key):
        # Case 1: Hit in T1 → promote to T2
        if key in self.t1:
            value = self.t1.pop(key)
            self.t2[key] = value
            self.t2.move_to_end(key)
            self.p = max(0, self.p - 1)
            self._evict_internal()
            return value
        # Case 2: Hit in T2 → refresh in T2
        if key in self.t2:
            value = self.t2.pop(key)
            self.t2[key] = value
            self.t2.move_to_end(key)
            self.p = min(self.maxsize, self.p + 1)
            self._evict_internal()
            return value
        # Case 3: Hit in B1 (ghost) → fetch and promote to T2
        if key in self.b1:
            self.b1.pop(key)
            self.p = min(self.maxsize, self.p + 1)
            self._evict_internal()
            value = super().__missing__(key)
            self.t2[key] = value
            self.t2.move_to_end(key)
            return value
        # Case 4: Hit in B2 (ghost) → fetch and promote to T2
        if key in self.b2:
            self.b2.pop(key)
            self.p = max(0, self.p - 1)
            self._evict_internal()
            value = super().__missing__(key)
            self.t2[key] = value
            self.t2.move_to_end(key)
            return value
        # Case 5: Cold miss → handled by Cache base class (calls __setitem__ after __missing__)
        return super().__getitem__(key)

    def __missing__(self, key):
        """
        Override this in a subclass, or rely on direct assignment (cache[key] = value).
        """
        raise KeyError(key)

    def pop(self, key, default=None):
        """
        Remove key from all lists.
        """
        for l in (self.t1, self.t2, self.b1, self.b2):
            if key in l:
                return l.pop(key)
        return default

    def clear(self):
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
