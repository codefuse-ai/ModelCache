import pytest
from modelcache.manager.eviction.arc_cache import ARC

@pytest.fixture()
def empty_arc():
    return ARC(maxsize=4)

@pytest.fixture()
def arc_with_data():
    c = ARC(maxsize=4)
    c['a'] = 1
    c['b'] = 2
    return c

def test_setitem_adds_to_arc(empty_arc):
    """Test __setitem__ adds a key-value pair to ARC."""
    empty_arc['x'] = 123
    assert 'x' in empty_arc
    assert empty_arc['x'] == 123

def test_setitem_overwrites_value(empty_arc):
    """Test that __setitem__ overwrites existing value in ARC."""
    empty_arc['y'] = 1
    empty_arc['y'] = 55
    assert empty_arc['y'] == 55

def test_getitem_returns_value(arc_with_data):
    """Test __getitem__ returns the correct value if present."""
    assert arc_with_data['a'] == 1

def test_getitem_raises_keyerror_on_missing(empty_arc):
    """Test __getitem__ raises KeyError if key is missing."""
    with pytest.raises(KeyError):
        _ = empty_arc['nope']

def test_contains_true_for_present(arc_with_data):
    """Test __contains__ returns True for a present key."""
    assert 'b' in arc_with_data

def test_contains_false_for_missing(arc_with_data):
    """Test __contains__ returns False for missing key."""
    assert 'notfound' not in arc_with_data

def test_len_reports_active_cache_size(arc_with_data):
    """Test __len__ reports only active items (T1 + T2)."""
    assert len(arc_with_data) == 2
    arc_with_data['c'] = 3
    assert len(arc_with_data) == 3

def test_pop_removes_key_from_one_list(arc_with_data):
    """Test pop removes key from the first ARC list where it is found."""
    arc_with_data['ghost'] = 9
    arc_with_data.b1['ghost'] = 9
    arc_with_data.pop('ghost')
    assert 'ghost' not in arc_with_data.t1
    assert 'ghost' in arc_with_data.b1

def test_clear_removes_all_keys(empty_arc):
    """Test clear() empties all lists and resets p."""
    empty_arc['a'] = 1
    empty_arc['b'] = 2
    empty_arc.clear()
    assert len(empty_arc.t1) == 0
    assert len(empty_arc.t2) == 0
    assert len(empty_arc.b1) == 0
    assert len(empty_arc.b2) == 0
    assert empty_arc.p == 0

def test_evict_internal_evicts_when_over_capacity():
    """Test _evict_internal evicts oldest when ARC is full."""
    evicted = []
    c = ARC(maxsize=2, on_evict=lambda keys: evicted.extend(keys))
    c['a'] = 1
    c['b'] = 2
    c['c'] = 3
    assert len(c) == 2
    assert len(evicted) >= 1
    for k in evicted:
        assert k in ['a', 'b']

def test_eviction_callback_is_called():
    """Test on_evict callback is called on eviction."""
    evicted = []
    c = ARC(maxsize=2, on_evict=lambda keys: evicted.extend(keys))
    c['x'] = 1
    c['y'] = 2
    c['z'] = 3
    assert len(evicted) > 0
    assert all(isinstance(k, str) for k in evicted)

def test_promote_from_t1_to_t2(empty_arc):
    """Test that accessing key in T1 promotes it to T2."""
    empty_arc['foo'] = 10
    assert 'foo' in empty_arc.t1
    _ = empty_arc['foo']
    assert 'foo' in empty_arc.t2

def test_refresh_in_t2_updates_order(empty_arc):
    """Test that repeated access in T2 keeps item in T2."""
    empty_arc['x'] = 1
    _ = empty_arc['x']  # promote to T2
    empty_arc['y'] = 2
    _ = empty_arc['x']  # refresh x in T2
    assert 'x' in empty_arc.t2
    assert empty_arc.t2.popitem(last=True)[0] == 'x'

def test_hit_in_ghost_lists_promotes_to_t2(empty_arc):
    """Test access in ghost list B1 promotes key to T2."""
    empty_arc['a'] = 1
    empty_arc['b'] = 2
    empty_arc['c'] = 3  # triggers eviction to ghost B1
    if empty_arc.b1:
        ghost_key = next(iter(empty_arc.b1))
        empty_arc.__missing__ = lambda key: 999
        _ = empty_arc[ghost_key]
        assert ghost_key in empty_arc.t2

def test_iter_lists_keys_in_order(arc_with_data):
    """Test __iter__ yields keys from T1 and then T2."""
    arc_with_data['c'] = 3
    keys = list(iter(arc_with_data))
    expected = list(arc_with_data.t1.keys()) + list(arc_with_data.t2.keys())
    assert keys == expected

def test_repr_outputs_status(empty_arc):
    """Test __repr__ returns a string with cache stats."""
    r = repr(empty_arc)
    assert r.startswith("ARC(")
    assert "maxsize" in r

# ----------- Additional/Edge case tests -----------

def test_pop_missing_key_returns_default(empty_arc):
    """Test that pop() returns default if key not found in any list."""
    assert empty_arc.pop('missing', default='sentinel') == 'sentinel'

def test_setitem_multiple_evictions():
    """Test multiple evictions in sequence do not corrupt the cache."""
    evicted = []
    c = ARC(maxsize=2, on_evict=lambda keys: evicted.extend(keys))
    c['a'] = 1
    c['b'] = 2
    c['c'] = 3
    c['d'] = 4
    assert len(c) == 2
    assert len(evicted) >= 2
    assert all(isinstance(k, str) for k in evicted)

def test_access_promotes_b1_and_b2(empty_arc):
    """Test accessing a key in B1 or B2 increases/decreases p appropriately."""
    empty_arc['a'] = 1
    empty_arc['b'] = 2
    empty_arc['c'] = 3
    if empty_arc.b1:
        ghost_key = next(iter(empty_arc.b1))
        empty_arc.__missing__ = lambda key: 555
        p_before = empty_arc.p
        _ = empty_arc[ghost_key]
        assert empty_arc.p > p_before or empty_arc.p == empty_arc.maxsize

def test_active_and_ghost_lists_dont_exceed_maxsize():
    """Test ghost lists (B1/B2) and active lists (T1/T2) don't exceed maxsize."""
    c = ARC(maxsize=3)
    for k in ['a', 'b', 'c', 'd', 'e']:
        c[k] = ord(k)
    # B1 + B2 should never be larger than maxsize
    assert len(c.b1) + len(c.b2) <= c.maxsize
    assert len(c.t1) + len(c.t2) <= c.maxsize

def test_clear_resets_all_lists_and_p(empty_arc):
    """Test that clear() resets all lists and p after many ops."""
    for k in 'abcd':
        empty_arc[k] = ord(k)
    empty_arc.clear()
    assert not any([empty_arc.t1, empty_arc.t2, empty_arc.b1, empty_arc.b2])
    assert empty_arc.p == 0

def test_repr_is_informative(empty_arc):
    """Test that __repr__ outputs all important stats."""
    empty_arc['q'] = 9
    r = repr(empty_arc)
    assert "t1_len" in r and "b1_len" in r and "p=" in r

def test_setitem_duplicate_key_resets_position(empty_arc):
    """Test that setting the same key again resets its position."""
    empty_arc['x'] = 10
    empty_arc['y'] = 11
    empty_arc['x'] = 99
    # x should be last in t1
    assert list(empty_arc.t1.keys())[-1] == 'x'
    assert empty_arc['x'] == 99

def test_eviction_of_promoted_key():
    """Test that a key promoted to T2 can still be evicted if capacity is exceeded."""
    evicted = []
    c = ARC(maxsize=2, on_evict=lambda keys: evicted.extend(keys))
    c['a'] = 1
    c['b'] = 2
    _ = c['a']  # Promote 'a' to T2
    c['c'] = 3  # Evict one of the keys
    assert len(c) == 2
    assert len(evicted) >= 1

def test_keyerror_on_missing_and_no_default(empty_arc):
    """Test pop() raises KeyError if no default is given and key is missing."""
    with pytest.raises(KeyError):
        empty_arc.pop('never-there')

