import pytest
import threading
import time
from modelcache.manager.eviction.wtinylfu_cache import W2TinyLFU, CountMinSketch

# ----------- Fixtures -----------

@pytest.fixture()
def empty_cache():
    # Returns an empty W2TinyLFU cache for tests
    return W2TinyLFU(maxsize=10, window_pct=50)

@pytest.fixture()
def cache_with_data():
    # Returns a W2TinyLFU cache with two preset items
    c = W2TinyLFU(maxsize=10, window_pct=50)
    c['a'] = 1
    c['b'] = 2
    return c

@pytest.fixture()
def cms():
    # Returns a CountMinSketch for CMS tests
    return CountMinSketch(width=16, depth=2, decay_interval=5)

# ----------- Basic Functionality -----------

def test_setitem_adds_to_cache(empty_cache):
    """Test that __setitem__ adds a new key-value pair to the cache."""
    empty_cache['x'] = 100
    assert empty_cache.data['x'] == 100
    assert 'x' in empty_cache

def test_setitem_overwrites_value(empty_cache):
    """Test that __setitem__ overwrites existing value for the same key."""
    empty_cache['y'] = 1
    empty_cache['y'] = 999
    assert empty_cache.data['y'] == 999

def test_getitem_returns_value(cache_with_data):
    """Test __getitem__ returns correct value for present key."""
    assert cache_with_data['a'] == 1

def test_getitem_raises_keyerror_on_missing(empty_cache):
    """Test __getitem__ raises KeyError for missing key."""
    with pytest.raises(KeyError):
        _ = empty_cache['not_found']

def test_get_method_returns_value(cache_with_data):
    """Test get method returns value for existing key."""
    assert cache_with_data.get('b') == 2

def test_get_method_returns_default_on_missing(empty_cache):
    """Test get method returns default for missing key."""
    assert empty_cache.get('zzz', default=123) == 123

def test_contains_true_for_present(cache_with_data):
    """Test __contains__ returns True for present key."""
    assert 'a' in cache_with_data

def test_contains_false_for_missing(cache_with_data):
    """Test __contains__ returns False for missing key."""
    assert 'zzz' not in cache_with_data

def test_delitem_removes_key(cache_with_data):
    """Test __delitem__ removes a present key."""
    del cache_with_data['a']
    assert 'a' not in cache_with_data

def test_delitem_safe_if_missing(empty_cache):
    """Test __delitem__ does not raise if key is missing."""
    try:
        del empty_cache['nope']
    except Exception:
        pytest.fail("delitem should not raise when key is missing.")

def test_duplicate_key_updates_value(empty_cache):
    """Test that setting same key twice updates the value."""
    empty_cache['dup'] = 10
    empty_cache['dup'] = 20
    assert empty_cache['dup'] == 20

def test_window_size_clamped_at_least_one():
    """Test that window size is always at least 1."""
    c = W2TinyLFU(maxsize=3, window_pct=0)
    assert c.window_size >= 1
    c['a'] = 1
    assert len(c.window) <= c.window_size

def test_clear_empties_cache(empty_cache):
    """Test that clear() empties the cache."""
    empty_cache['x'] = 1
    empty_cache['y'] = 2
    empty_cache.clear()
    assert len(empty_cache.data) == 0
    assert list(empty_cache.window) == []
    assert list(empty_cache.probation) == []
    assert list(empty_cache.protected) == []

# ----------- Eviction and Frequency -----------

def test_evicted_key_not_accessible():
    """Test that accessing an evicted key raises KeyError."""
    c = W2TinyLFU(maxsize=2)
    c['a'], c['b'] = 1, 2
    c['c'] = 3
    # Only two elements can be present, so one must be gone
    with pytest.raises(KeyError):
        # Either 'a' or 'b' must be gone; test both if needed
        try:
            _ = c['a']
        except KeyError:
            _ = c['b']

def test_eviction_until_empty_and_reusability():
    """Test cache remains usable after all items have been evicted."""
    c = W2TinyLFU(maxsize=2)
    c['x'] = 1
    c['y'] = 2
    c['z'] = 3
    del c['y']
    c['w'] = 4
    assert len(c.data) <= 2

def test_high_frequency_key_survives_eviction():
    """Test that a frequently accessed key survives eviction pressure."""
    c = W2TinyLFU(maxsize=10, window_pct=40)
    c['a'] = 1
    for new_key in ['b', 'c', 'd', 'e', 'f', 'g']:
        c.get('a')
        c[new_key] = ord(new_key)
    assert 'a' in c

def test_low_freq_key_evicted():
    """Test that an infrequently used key is evicted under pressure."""
    c = W2TinyLFU(maxsize=3, window_pct=33)
    c['a'] = 1
    c['b'] = 2
    c['c'] = 3
    c['d'] = 4
    # At least one of the early keys must be gone (usually 'a')
    evicted = {'a', 'b', 'c'} - set([k for k in c])
    assert len(evicted) >= 1

@pytest.mark.parametrize("hits, expected_in_cache", [
    (0, False),
    (5, True),
    (15, True),
])
def test_access_promotes_key(hits, expected_in_cache):
    """Test that frequent access promotes key to survive eviction."""
    c = W2TinyLFU(maxsize=8, window_pct=40)
    c['a'] = 1
    insert_keys = ['b', 'c', 'd', 'e', 'f', 'g', 'h']
    for i, k in enumerate(insert_keys):
        if i < hits:
            c.get('a')
        c[k] = ord(k)
    assert ('a' in c) == expected_in_cache

def test_cms_estimate_increases_and_decays(cms):
    """Test that CountMinSketch increases and decays correctly."""
    before = cms.estimate('test')
    cms.add('test')
    after_add = cms.estimate('test')
    assert after_add >= before
    cms.decay()
    after_decay = cms.estimate('test')
    assert after_decay <= after_add

# ----------- Segment and Admission Logic -----------

def test_admit_to_main_adds_to_probation():
    """Test _admit_to_main adds new key to probation segment."""
    c = W2TinyLFU(maxsize=10, window_pct=50)
    key = 'k'
    assert key not in c.probation
    assert key not in c.protected
    c._admit_to_main(key)
    assert key in c.probation

def test_admit_to_main_evicts_when_probation_full():
    """Test _admit_to_main evicts LRU key when probation is full."""
    c = W2TinyLFU(maxsize=4, window_pct=25)
    for i in range(c.probation_size):
        c._admit_to_main(f'p{i}')
        c.data[f'p{i}'] = i
    extra = 'extra'
    c.data[extra] = 999
    old_probation_keys = set(c.probation.keys())
    c._admit_to_main(extra)
    assert len(c.probation) == c.probation_size
    assert extra in c.probation
    assert len(old_probation_keys - set(c.probation.keys())) >= 1

def test_admit_to_main_noop_if_already_present():
    """Test _admit_to_main does nothing if key already present."""
    c = W2TinyLFU(maxsize=10, window_pct=50)
    key = 'present'
    c.probation[key] = True
    c._admit_to_main(key)
    assert key in c.probation
    c.protected[key] = True
    c._admit_to_main(key)
    assert key in c.protected

def test_put_key_in_window_and_main():
    """Test that new keys are distributed into segments correctly."""
    c = W2TinyLFU(maxsize=10, window_pct=50)
    c['a'] = 1
    for k in ['b', 'c', 'd']:
        c[k] = ord(k)
    c['e'] = 5
    total = len(c.window) + len(c.probation) + len(c.protected)
    assert total <= c.maxsize

# ----------- Extra Edge Cases -----------

def test_clear_on_empty_cache():
    """Test clear() on an already empty cache does not fail."""
    c = W2TinyLFU(maxsize=3)
    c.clear()
    assert len(c.data) == 0

def test_get_returns_default_if_not_present():
    """Test get() returns default if the key is missing."""
    c = W2TinyLFU(maxsize=3)
    assert c.get('notfound', default=777) == 777

def test_cache_survives_rapid_inserts_and_deletes():
    """Test cache remains consistent under rapid inserts/deletes."""
    c = W2TinyLFU(maxsize=3)
    for i in range(30):
        c[f'k{i%3}'] = i
        if i % 2 == 0:
            del c[f'k{(i+1)%3}']
    assert len(c.data) <= 3

def test_cache_never_exceeds_maxsize():
    """Test that cache never exceeds its declared maxsize."""
    c = W2TinyLFU(maxsize=5)
    for i in range(20):
        c[f'x{i}'] = i
    assert len(c.window) + len(c.probation) + len(c.protected) <= 5

# ----------- Concurrency Tests -----------

def test_concurrent_setitem_and_getitem():
    """Test concurrent __setitem__ and __getitem__ do not corrupt cache state."""
    cache = W2TinyLFU(maxsize=8, window_pct=50)
    keys = [f'k{i}' for i in range(16)]
    exceptions = []

    def writer():
        for k in keys:
            try:
                cache[k] = ord(k[-1])
                time.sleep(0.001)
            except Exception as e:
                exceptions.append(e)

    def reader():
        for _ in range(20):
            for k in keys:
                try:
                    _ = cache.get(k, None)
                except Exception as e:
                    exceptions.append(e)
            time.sleep(0.001)

    threads = [threading.Thread(target=writer)] + [threading.Thread(target=reader) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not exceptions, f"Exceptions in threads: {exceptions}"

def test_concurrent_delitem_and_setitem():
    """Test concurrent __delitem__ and __setitem__ do not cause errors or corruption."""
    cache = W2TinyLFU(maxsize=6, window_pct=50)
    for k in range(6):
        cache[f'x{k}'] = k

    del_exceptions = []
    set_exceptions = []

    def deleter():
        for _ in range(12):
            try:
                del cache[f'x{_ % 6}']
            except KeyError:
                pass  # Acceptable: may not be present
            except Exception as e:
                del_exceptions.append(e)
            time.sleep(0.001)

    def setter():
        for i in range(12, 24):
            try:
                cache[f'x{i % 6}'] = i
            except Exception as e:
                set_exceptions.append(e)
            time.sleep(0.001)

    t1 = threading.Thread(target=deleter)
    t2 = threading.Thread(target=setter)
    t1.start()
    t2.start()
    t1.join()
    t2.join()
    assert not del_exceptions, f"Exceptions in deleter: {del_exceptions}"
    assert not set_exceptions, f"Exceptions in setter: {set_exceptions}"

def test_concurrent_iterators_with_modifications():
    """Test that iterating over keys while modifying the cache does not throw."""
    cache = W2TinyLFU(maxsize=7, window_pct=50)
    for i in range(7):
        cache[f'k{i}'] = i

    iter_exceptions = []

    def iterate():
        try:
            for _ in range(10):
                list(cache.data.keys())
                time.sleep(0.002)
        except Exception as e:
            iter_exceptions.append(e)

    def modifier():
        for i in range(10, 20):
            cache[f'k{i%7}'] = i
            time.sleep(0.001)

    t1 = threading.Thread(target=iterate)
    t2 = threading.Thread(target=modifier)
    t1.start()
    t2.start()
    t1.join()
    t2.join()
    assert not iter_exceptions, f"Exceptions during concurrent iteration: {iter_exceptions}"

def test_concurrent_clear_and_get():
    """Test that clear and get do not deadlock or throw under concurrent access."""
    cache = W2TinyLFU(maxsize=10, window_pct=50)
    for i in range(10):
        cache[f'c{i}'] = i

    clear_exceptions = []
    get_exceptions = []

    def clearer():
        for _ in range(5):
            try:
                cache.clear()
                time.sleep(0.005)
            except Exception as e:
                clear_exceptions.append(e)

    def getter():
        for _ in range(15):
            for i in range(10):
                try:
                    _ = cache.get(f'c{i}', None)
                except Exception as e:
                    get_exceptions.append(e)
            time.sleep(0.001)

    t1 = threading.Thread(target=clearer)
    t2 = threading.Thread(target=getter)
    t1.start()
    t2.start()
    t1.join()
    t2.join()
    assert not clear_exceptions, f"Exceptions: {clear_exceptions}"
    assert not get_exceptions, f"Exceptions: {get_exceptions}"

def test_lock_allows_multiple_readers_but_exclusive_writer():
    """Test rwlock allows multiple readers at once but only one writer at a time."""
    cache = W2TinyLFU(maxsize=4, window_pct=50)
    shared_counter = [0]
    read_count = []
    write_count = []

    def reader():
        with cache._rw_lock.gen_rlock():
            val = shared_counter[0]
            time.sleep(0.002)
            read_count.append(val)

    def writer():
        with cache._rw_lock.gen_wlock():
            current = shared_counter[0]
            shared_counter[0] = current + 1
            time.sleep(0.003)
            write_count.append(shared_counter[0])

    threads = []
    for _ in range(3):
        threads.append(threading.Thread(target=reader))
    threads.append(threading.Thread(target=writer))
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert any(r == read_count[0] for r in read_count[1:]), "Multiple readers did not overlap"
    assert len(write_count) == 1

