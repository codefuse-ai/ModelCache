import pytest
from modelcache.manager.eviction.wtinylfu_cache import W2TinyLFU, CountMinSketch

@pytest.fixture()
def empty_cache():
    return W2TinyLFU(maxsize=10, window_pct=50)

@pytest.fixture()
def cache_with_data():
    c = W2TinyLFU(maxsize=10, window_pct=50)
    c['a'] = 1
    c['b'] = 2
    return c

@pytest.fixture()
def cms():
    return CountMinSketch(width=16, depth=2, decay_interval=5)

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

def test_eviction_callback_at_least_one_eviction():
    """Test that eviction callback is called when capacity exceeded."""
    evicted = []
    c = W2TinyLFU(maxsize=3, window_pct=33, on_evict=evicted.append)
    c['a'] = 1
    c['b'] = 2
    c['c'] = 3
    c['d'] = 4
    assert len(evicted) >= 1
    assert all(k in ('a', 'b', 'c', 'd') for k in evicted)

def test_eviction_callback_evicted_keys_are_gone():
    """Test that evicted keys are not present in the cache."""
    evicted = []
    c = W2TinyLFU(maxsize=3, window_pct=33, on_evict=evicted.append)
    keys = ['a', 'b', 'c', 'd']
    for i, k in enumerate(keys):
        c[k] = i
    for k in evicted:
        assert k not in c

def test_eviction_callback_not_called_when_under_capacity():
    """Test that eviction callback is not called when under capacity."""
    evicted = []
    c = W2TinyLFU(maxsize=4, window_pct=50, on_evict=evicted.append)
    c['one'] = 1
    c['two'] = 2
    assert len(evicted) == 0

def test_eviction_callback_with_chain_eviction():
    """Test that multiple evictions can occur in chain scenarios."""
    evicted = []
    c = W2TinyLFU(maxsize=2, window_pct=50, on_evict=evicted.append)
    c['x'] = 1
    c['y'] = 2
    c['z'] = 3
    c['w'] = 4
    assert set(evicted).issubset({'x', 'y', 'z', 'w'})
    assert len(evicted) >= 1

def test_evicted_key_not_accessible():
    """Test that accessing an evicted key raises KeyError."""
    c = W2TinyLFU(maxsize=2)
    c['a'], c['b'] = 1, 2
    c['c'] = 3
    with pytest.raises(KeyError):
        _ = c['a']

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
    assert 'a' not in c

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
    if hits == 0:
        assert 'a' not in c
    else:
        assert 'a' in c

def test_cms_estimate_increases_and_decays(cms):
    """Test that CountMinSketch increases and decays correctly."""
    before = cms.estimate('test')
    cms.add('test')
    after_add = cms.estimate('test')
    assert after_add >= before
    cms.decay()
    after_decay = cms.estimate('test')
    assert after_decay <= after_add

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
    evicted = []
    c = W2TinyLFU(maxsize=4, window_pct=25, on_evict=evicted.append)
    for i in range(c.probation_size):
        c._admit_to_main(f'p{i}')
        c.data[f'p{i}'] = i
    extra = 'extra'
    c.data[extra] = 999
    c._admit_to_main(extra)
    assert len(c.probation) == c.probation_size
    assert extra in c.probation
    assert len(evicted) >= 1

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

def test_put_eviction_callback_called():
    """Test that the eviction callback is invoked when needed."""
    evicted = []
    c = W2TinyLFU(maxsize=2, on_evict=evicted.append)
    c['a'] = 1
    c['b'] = 2
    c['c'] = 3
    assert len(evicted) >= 1
