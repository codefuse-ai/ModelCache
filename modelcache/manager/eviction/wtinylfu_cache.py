from cachetools import LRUCache, Cache, LFUCache
from readerwriterlock import rwlock
import random

class CountMinSketch:
    def __init__(self, width=1024, depth=4, decay_interval=10000):
        """Initialize Count-Min Sketch with specified dimensions."""
        self.width = width
        self.depth = depth
        self.tables = [[0]*width for _ in range(depth)]  # Hash tables
        self.seeds = [random.randrange(1<<30) for _ in range(depth)]  # Hash seeds
        self.ops = 0  # Operation counter for decay trigger
        self.decay_interval = decay_interval

    def _hash(self, x, seed):
        """Hash function for mapping items to table positions."""
        return hash((x, seed)) % self.width

    def add(self, x):
        """Add an item and increment its frequency estimate."""
        self.ops += 1
        est = self.estimate(x)  # Get current estimate
        # Update all hash tables
        for i, seed in enumerate(self.seeds):
            idx = self._hash(x, seed)
            if self.tables[i][idx] <= est:
                self.tables[i][idx] += 1

        # Periodic decay to handle changing patterns
        if self.ops >= self.decay_interval:
            self.decay()
            self.ops = 0

    def estimate(self, x):
        """Estimate frequency of an item (minimum across all tables)."""
        return min(self.tables[i][self._hash(x, seed)]
                   for i, seed in enumerate(self.seeds))

    def decay(self):
        """Decay all frequency counts by half."""
        for table in self.tables:
            for i in range(len(table)):
                table[i] >>= 1  # Right shift (divide by 2)

class W2TinyLFU(Cache):
    """
    Window Tiny LFU cache implementation.

    Combines a small LRU window cache with a main cache divided into
    probation and protected segments, using frequency estimation for
    admission control.
    """

    def __init__(self, maxsize, window_pct=0.01):
        """
        Initialize W-TinyLFU cache.

        Args:
            maxsize: Maximum size of the cache
            window_pct: Percentage of cache size for the window (default 1%)
        """
        super().__init__(maxsize)
        self.window_size = max(1, int(maxsize * window_pct))
        rest = maxsize - self.window_size
        self.probation_size = rest // 2
        self.protected_size = rest - self.probation_size

        # Three cache segments
        self.window = LRUCache(maxsize=self.window_size)       # Recent items
        self.probation = LFUCache(maxsize=self.probation_size) # New main cache items
        self.protected = LFUCache(maxsize=self.protected_size) # Frequently accessed items

        self.cms = CountMinSketch()  # Frequency estimator
        self.data = {}  # Cache data storage
        self._rw_lock = rwlock.RWLockWrite()  # Read-write lock for thread safety

    def __setitem__(self, key, value):
        """Add or update an item in the cache."""
        with self._rw_lock.gen_wlock():
            self.data[key] = value
            self._put(key)

    def __getitem__(self, key):
        """Retrieve an item from the cache."""
        val = self.get(key, default=None)
        if val is None:
            raise KeyError(key)
        return val

    def __contains__(self, key):
        """Check if an item exists in the cache."""
        return key in self.window or key in self.probation or key in self.protected

    def __delitem__(self, key):
        """Remove an item from the cache."""
        with self._rw_lock.gen_wlock():
            self.data.pop(key, None)
            self.window.pop(key, None)
            self.probation.pop(key, None)
            self.protected.pop(key, None)

    def get(self, key, default=None):
        """
        Retrieve an item from the cache, updating its position
        in the cache hierarchy if necessary.
        """
        if key in self.window:
            self.window[key] = True
            return self.data.get(key, default)
        if key in self.protected:
            self.protected[key] = True
            return self.data.get(key, default)
        if key in self.probation:
            self.probation.pop(key)
            if len(self.protected) >= self.protected_size:
                demoted = next(iter(self.protected))
                self.protected.pop(demoted)
                self.probation[demoted] = True
            self.protected[key] = True
            return self.data.get(key, default)
        return default

    def _put(self, key):
        """
        Add an item to the cache, using frequency-based admission
        control and eviction policies.
        """
        self.cms.add(key)
        if key in self:
            return

        if len(self.window) < self.window_size:
            self.window[key] = True
            return

        victim = next(iter(self.window))
        self.window.pop(victim)

        if self.cms.estimate(key) >= self.cms.estimate(victim):
            self._admit_to_main(victim)
            self._admit_to_main(key)
        else:
            self._admit_to_main(victim)
            self.data.pop(key, None)

    def _admit_to_main(self, key):
        """
        Admit an item to the main cache (probation or protected segment).
        """
        if key in self.protected or key in self.probation:
            return
        if self.probation_size == 0:
            self.data.pop(key, None)
            return
        if len(self.probation) < self.probation_size:
            self.probation[key] = True
        elif self.probation:
            evicted = next(iter(self.probation))
            self.probation.pop(evicted)
            self.probation[key] = True
            self.data.pop(evicted, None)
        else:
            self.data.pop(key, None)

    def clear(self):
        """Clear all items from the cache."""
        with self._rw_lock.gen_wlock():
            self.window.clear()
            self.probation.clear()
            self.protected.clear()
            self.data.clear()

