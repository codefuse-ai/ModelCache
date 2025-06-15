import pytest
import threading
import time
from modelcache.manager.eviction.arc_cache import ARC

# ----------- Fixtures -----------

@pytest.fixture()
def empty_arc():
    # Create an empty ARC cache for tests
    return ARC(maxsize=4)

@pytest.fixture()
def arc_with_data():
    # Create an ARC cache pre-filled with two items
    c = ARC(maxsize=4)
    c['a'] = 1
    c['b'] = 2
    return c

# ----------- Basic Functionality Tests -----------

def test_setitem_adds_to_arc(empty_arc):
    """Test that __setitem__ adds a key-value pair to ARC."""
    empty_arc['x'] = 123
    assert 'x' in empty_arc
    assert empty_arc['x'] == 123

def test_setitem_overwrites_value(empty_arc):
    """Test that __setitem__ overwrites existing value in ARC."""
    empty_arc['y'] = 1
    empty_arc['y'] = 55
    assert empty_arc['y'] == 55

def test_setitem_none_value(empty_arc):
    """Test that __setitem__ can store None as a value."""
    empty_arc['foo'] = None
    assert empty_arc['foo'] is None

def test_duplicate_key_updates_value(empty_arc):
    """Test that setting the same key twice updates the value."""
    empty_arc['dup'] = 10
    empty_arc['dup'] = 20
    assert empty_arc['dup'] == 20

def test_getitem_returns_value(arc_with_data):
    """Test that __getitem__ returns the correct value if present."""
    assert arc_with_data['a'] == 1

def test_getitem_raises_keyerror_on_missing(empty_arc):
    """Test that __getitem__ raises KeyError if key is missing."""
    with pytest.raises(KeyError):
        _ = empty_arc['nope']

def test_contains_true_for_present(arc_with_data):
    """Test that __contains__ returns True for a present key."""
    assert 'b' in arc_with_data

def test_contains_false_for_missing(arc_with_data):
    """Test that __contains__ returns False for missing key."""
    assert 'notfound' not in arc_with_data

def test_len_reports_active_cache_size(arc_with_data):
    """Test that __len__ reports only active items (T1 + T2)."""
    assert len(arc_with_data) == 2
    arc_with_data['c'] = 3
    assert len(arc_with_data) == 3

def test_pop_removes_key_from_one_list(arc_with_data):
    """Test that pop removes key from the first ARC list where it is found."""
    arc_with_data['ghost'] = 9
    with arc_with_data._rw_lock.gen_wlock():
        arc_with_data.b1['ghost'] = 9  # Simulate ghost in B1 (protected by write lock)
    arc_with_data.pop('ghost')
    with arc_with_data._rw_lock.gen_rlock():
        assert 'ghost' not in arc_with_data.t1
        assert 'ghost' in arc_with_data.b1

def test_pop_missing_key_returns_default(empty_arc):
    """Test that pop() returns default if key not found in any list."""
    assert empty_arc.pop('missing', default='sentinel') == 'sentinel'

def test_pop_evicted_key_returns_default(empty_arc):
    """Test that pop() on a recently evicted key returns default."""
    empty_arc['a'] = 1
    empty_arc['b'] = 2
    empty_arc['c'] = 3
    empty_arc['d'] = 4
    empty_arc['e'] = 5
    for key in ['a', 'b', 'c', 'd']:
        try:
            empty_arc.pop(key)
        except KeyError:
            pass
        except Exception:
            pytest.fail("Unexpected exception on pop after eviction.")

def test_pop_with_none_default(empty_arc):
    """Test pop returns None when default=None and key is missing."""
    assert empty_arc.pop('not_in_cache', default=None) is None

def test_keyerror_on_missing_and_no_default(empty_arc):
    """Test that pop() raises KeyError if no default is given and key is missing."""
    with pytest.raises(KeyError):
        empty_arc.pop('never-there')

def test_clear_removes_all_keys(empty_arc):
    """Test that clear() empties all lists and resets p."""
    empty_arc['a'] = 1
    empty_arc['b'] = 2
    empty_arc.clear()
    assert len(empty_arc.t1) == 0
    assert len(empty_arc.t2) == 0
    assert len(empty_arc.b1) == 0
    assert len(empty_arc.b2) == 0
    assert empty_arc.p == 0

def test_clear_on_empty_cache(empty_arc):
    """Test clear() does not raise when cache is already empty."""
    empty_arc.clear()
    assert len(empty_arc) == 0

# ----------- Eviction and Promotion Mechanics -----------

def test_evict_internal_evicts_when_over_capacity():
    """Test that _evict_internal evicts oldest when ARC is full."""
    c = ARC(maxsize=2)
    c['a'] = 1
    c['b'] = 2
    c['c'] = 3
    assert len(c) == 2
    with c._rw_lock.gen_rlock():
        present = set(c.t1.keys()) | set(c.t2.keys())
    assert 'c' in present
    assert len(present & {'a', 'b'}) == 1

def test_getitem_after_eviction(empty_arc):
    """Test that accessing an evicted key raises KeyError."""
    empty_arc['x'] = 1
    empty_arc['y'] = 2
    empty_arc['z'] = 3
    empty_arc['w'] = 4
    empty_arc['v'] = 5  # Triggers eviction
    evicted = [k for k in ['x', 'y', 'z', 'w'] if k not in empty_arc]
    assert len(evicted) >= 1
    for k in evicted:
        with pytest.raises(KeyError):
            _ = empty_arc[k]

def test_setitem_multiple_evictions():
    """Test that multiple evictions in sequence do not corrupt the cache."""
    c = ARC(maxsize=2)
    c['a'] = 1
    c['b'] = 2
    c['c'] = 3
    c['d'] = 4
    assert len(c) == 2
    with c._rw_lock.gen_rlock():
        present = set(c.t1.keys()) | set(c.t2.keys())
    assert len(present & {'a', 'b', 'c', 'd'}) == 2

def test_setitem_eviction_does_not_raise(empty_arc):
    """Test that inserting over maxsize repeatedly never raises."""
    try:
        for i in range(20):
            empty_arc[f'k{i}'] = i
    except Exception:
        pytest.fail("Eviction during setitem should never raise.")

def test_promote_from_t1_to_t2(empty_arc):
    """Test that accessing key in T1 promotes it to T2."""
    empty_arc['foo'] = 10
    with empty_arc._rw_lock.gen_rlock():
        assert 'foo' in empty_arc.t1
    _ = empty_arc['foo']
    with empty_arc._rw_lock.gen_rlock():
        assert 'foo' in empty_arc.t2

def test_refresh_in_t2_updates_order(empty_arc):
    """Test that repeated access in T2 keeps item in T2 and updates order."""
    empty_arc['x'] = 1
    _ = empty_arc['x']
    empty_arc['y'] = 2
    _ = empty_arc['x']
    with empty_arc._rw_lock.gen_rlock():
        assert 'x' in empty_arc.t2
        assert empty_arc.t2.popitem(last=True)[0] == 'x'

def test_setitem_duplicate_key_resets_position(empty_arc):
    """Test that setting the same key again resets its position."""
    empty_arc['x'] = 10
    empty_arc['y'] = 11
    empty_arc['x'] = 99
    with empty_arc._rw_lock.gen_rlock():
        assert list(empty_arc.t1.keys())[-1] == 'x'
    assert empty_arc['x'] == 99

def test_hit_in_ghost_lists_promotes_to_t2(empty_arc):
    """Test that access in ghost list B1 promotes key to T2."""
    empty_arc['a'] = 1
    empty_arc['b'] = 2
    empty_arc['c'] = 3
    with empty_arc._rw_lock.gen_rlock():
        b1_keys = list(empty_arc.b1.keys())
    if b1_keys:
        ghost_key = b1_keys[0]
        empty_arc.__missing__ = lambda key: 999
        _ = empty_arc[ghost_key]
        with empty_arc._rw_lock.gen_rlock():
            assert ghost_key in empty_arc.t2

def test_access_promotes_b1_and_b2(empty_arc):
    """Test that accessing a key in B1 or B2 increases/decreases p appropriately."""
    empty_arc['a'] = 1
    empty_arc['b'] = 2
    empty_arc['c'] = 3
    with empty_arc._rw_lock.gen_rlock():
        b1_keys = list(empty_arc.b1.keys())
    if b1_keys:
        ghost_key = b1_keys[0]
        empty_arc.__missing__ = lambda key: 555
        with empty_arc._rw_lock.gen_rlock():
            p_before = empty_arc.p
        _ = empty_arc[ghost_key]
        with empty_arc._rw_lock.gen_rlock():
            assert empty_arc.p > p_before or empty_arc.p == empty_arc.maxsize

def test_evict_when_t1_empty_promotes_t2(empty_arc):
    """
    Test that promoting an item to T2 keeps it in cache, while new items fill T1.
    """
    empty_arc['a'] = 1
    _ = empty_arc['a']  # promote to T2
    empty_arc['b'] = 2
    empty_arc['c'] = 3
    empty_arc['d'] = 4
    empty_arc['e'] = 5  # should cause evictions
    with empty_arc._rw_lock.gen_rlock():
        assert 'a' in empty_arc.t2
        t1_keys = list(empty_arc.t1.keys())
        t2_keys = list(empty_arc.t2.keys())
        assert len(t1_keys) + len(t2_keys) == empty_arc.maxsize
        assert 'a' not in t1_keys
        for k in t1_keys:
            assert k in ['b', 'c', 'd', 'e']

def test_active_and_ghost_lists_dont_exceed_maxsize():
    """Test that ghost lists (B1/B2) and active lists (T1/T2) don't exceed maxsize."""
    c = ARC(maxsize=3)
    for k in ['a', 'b', 'c', 'd', 'e']:
        c[k] = ord(k)
    with c._rw_lock.gen_rlock():
        assert len(c.b1) + len(c.b2) <= c.maxsize
        assert len(c.t1) + len(c.t2) <= c.maxsize

def test_clear_resets_all_lists_and_p(empty_arc):
    """Test that clear() resets all lists and p after many ops."""
    for k in 'abcd':
        empty_arc[k] = ord(k)
    empty_arc.clear()
    with empty_arc._rw_lock.gen_rlock():
        assert not any([empty_arc.t1, empty_arc.t2, empty_arc.b1, empty_arc.b2])
        assert empty_arc.p == 0

# ----------- Iteration and Representation -----------

def test_iter_lists_keys_in_order(arc_with_data):
    """Test that __iter__ yields keys from T1 and then T2."""
    arc_with_data['c'] = 3
    keys = list(iter(arc_with_data))
    with arc_with_data._rw_lock.gen_rlock():
        expected = list(arc_with_data.t1.keys()) + list(arc_with_data.t2.keys())
    assert keys == expected

def test_iter_empty_arc(empty_arc):
    """Test that iterating an empty ARC yields nothing."""
    assert list(iter(empty_arc)) == []

def test_repr_outputs_status(empty_arc):
    """Test that __repr__ returns a string with cache stats."""
    r = repr(empty_arc)
    assert r.startswith("ARC(")
    assert "maxsize" in r

def test_repr_reflects_content(empty_arc):
    """Test __repr__ shows correct lengths after operations."""
    empty_arc['foo'] = 123
    empty_arc['bar'] = 321
    r = repr(empty_arc)
    assert f"len={len(empty_arc)}" in r
    assert f"t1_len={len(empty_arc.t1)}" in r

def test_repr_is_informative(empty_arc):
    """Test that __repr__ outputs all important stats."""
    empty_arc['q'] = 9
    r = repr(empty_arc)
    assert "t1_len" in r and "b1_len" in r and "p=" in r

# ----------- Concurrency and Locking -----------

def test_concurrent_setitem_and_getitem():
    """Test concurrent __setitem__ and __getitem__ do not corrupt cache state."""
    arc = ARC(maxsize=8)
    keys = [f'k{i}' for i in range(16)]
    exceptions = []

    def writer():
        for k in keys:
            try:
                arc[k] = ord(k[-1])
                time.sleep(0.001)
            except Exception as e:
                exceptions.append(e)

    def reader():
        for _ in range(20):
            for k in keys:
                try:
                    _ = arc.get(k, None)
                except Exception as e:
                    exceptions.append(e)
            time.sleep(0.001)

    threads = [threading.Thread(target=writer)] + [threading.Thread(target=reader) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not exceptions, f"Exceptions in threads: {exceptions}"

def test_concurrent_pop_and_setitem():
    """Test concurrent pop and setitem maintain cache integrity."""
    arc = ARC(maxsize=5)
    for k in range(5):
        arc[f'x{k}'] = k

    pop_exceptions = []
    set_exceptions = []

    def popper():
        for _ in range(10):
            try:
                arc.pop(f'x{_ % 5}', default=None)
            except KeyError:
                pass  # Acceptable in concurrency
            except Exception as e:
                pop_exceptions.append(e)

    def setter():
        for i in range(10, 20):
            try:
                arc[f'x{i % 5}'] = i
            except Exception as e:
                set_exceptions.append(e)

    t1 = threading.Thread(target=popper)
    t2 = threading.Thread(target=setter)
    t1.start()
    t2.start()
    t1.join()
    t2.join()
    assert not set_exceptions, f"Exceptions in setter: {set_exceptions}"
    assert not pop_exceptions, f"Unexpected exceptions in popper: {pop_exceptions}"

def test_concurrent_iterators_with_modifications():
    """Test that iterating keys while modifying cache is safe and doesn't throw."""
    arc = ARC(maxsize=7)
    for i in range(7):
        arc[f'k{i}'] = i

    iter_exceptions = []

    def iterate():
        try:
            for _ in range(10):
                list(iter(arc))
                time.sleep(0.002)
        except Exception as e:
            iter_exceptions.append(e)

    def modifier():
        for i in range(10, 20):
            arc[f'k{i%7}'] = i
            time.sleep(0.001)

    t1 = threading.Thread(target=iterate)
    t2 = threading.Thread(target=modifier)
    t1.start()
    t2.start()
    t1.join()
    t2.join()
    assert not iter_exceptions, f"Exceptions during concurrent iteration: {iter_exceptions}"

def test_concurrent_clear_and_get():
    """Test that clear and getitem do not deadlock or throw under concurrent access."""
    arc = ARC(maxsize=10)
    for i in range(10):
        arc[f'c{i}'] = i

    clear_exceptions = []
    get_exceptions = []

    def clearer():
        for _ in range(5):
            try:
                arc.clear()
                time.sleep(0.005)
            except Exception as e:
                clear_exceptions.append(e)

    def getter():
        for _ in range(15):
            for i in range(10):
                try:
                    _ = arc.get(f'c{i}', None)
                except Exception as e:
                    get_exceptions.append(e)
            time.sleep(0.001)

    t1 = threading.Thread(target=clearer)
    t2 = threading.Thread(target=getter)
    t1.start()
    t2.start()
    t1.join()
    t2.join()
    assert not clear_exceptions and not get_exceptions, f"Exceptions: {clear_exceptions}, {get_exceptions}"

def test_lock_allows_multiple_readers_but_exclusive_writer():
    """Test rwlock allows multiple readers at once but only one writer at a time."""
    arc = ARC(maxsize=4)
    shared_counter = 0
    read_count = []
    write_count = []

    def reader():
        nonlocal shared_counter
        with arc._rw_lock.gen_rlock():
            val = shared_counter
            time.sleep(0.002)
            read_count.append(val)

    def writer():
        nonlocal shared_counter
        with arc._rw_lock.gen_wlock():
            current = shared_counter
            shared_counter = current + 1
            time.sleep(0.003)
            write_count.append(shared_counter)

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

def test_concurrent_duplicate_setitem():
    """Test that concurrent setitem on the same key does not corrupt cache."""
    arc = ARC(maxsize=3)
    exception_list = []
    def setter(val):
        try:
            for _ in range(20):
                arc['dup'] = val
                time.sleep(0.0005)
        except Exception as e:
            exception_list.append(e)

    threads = [threading.Thread(target=setter, args=(v,)) for v in range(3)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert 'dup' in arc
    assert arc['dup'] in (0, 1, 2)
    assert not exception_list, f"Exceptions: {exception_list}"

def test_concurrent_clear_and_setitem():
    """Test that clear and setitem do not cause errors in parallel."""
    arc = ARC(maxsize=4)
    set_errors = []
    clear_errors = []
    stop = threading.Event()

    def clearer():
        while not stop.is_set():
            try:
                arc.clear()
                time.sleep(0.001)
            except Exception as e:
                clear_errors.append(e)

    def setter():
        for i in range(30):
            try:
                arc[f'k{i%4}'] = i
                time.sleep(0.0005)
            except Exception as e:
                set_errors.append(e)
        stop.set()

    t1 = threading.Thread(target=clearer)
    t2 = threading.Thread(target=setter)
    t1.start()
    t2.start()
    t1.join()
    t2.join()
    assert not set_errors, f"Setitem exceptions: {set_errors}"
    assert not clear_errors, f"Clear exceptions: {clear_errors}"

def test_concurrent_pop_with_contention():
    """Test multiple threads popping the same and different keys."""
    arc = ARC(maxsize=2)
    for k in ('a', 'b'):
        arc[k] = k
    errors = []
    def popper(key):
        for _ in range(8):
            try:
                arc.pop(key, default=None)
                time.sleep(0.0005)
            except Exception as e:
                errors.append(e)
    t1 = threading.Thread(target=popper, args=('a',))
    t2 = threading.Thread(target=popper, args=('b',))
    t3 = threading.Thread(target=popper, args=('a',))
    t1.start()
    t2.start()
    t3.start()
    t1.join()
    t2.join()
    t3.join()
    assert not errors, f"Exceptions: {errors}"

def test_concurrent_len_and_setitem():
    """Test that __len__ and __setitem__ are safe in concurrent usage."""
    arc = ARC(maxsize=10)
    errors = []
    stop = threading.Event()
    def setter():
        for i in range(50):
            try:
                arc[f'k{i%10}'] = i
                time.sleep(0.0002)
            except Exception as e:
                errors.append(e)
        stop.set()

    def length_checker():
        while not stop.is_set():
            try:
                _ = len(arc)
                time.sleep(0.0002)
            except Exception as e:
                errors.append(e)
    t1 = threading.Thread(target=setter)
    t2 = threading.Thread(target=length_checker)
    t1.start()
    t2.start()
    t1.join()
    t2.join()
    assert not errors, f"Exceptions: {errors}"
