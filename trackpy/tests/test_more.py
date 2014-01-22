import numpy as np
from trackpy.tracking import PointND, link, Hash_table
from copy import deepcopy

# Call lambda function for a fresh copy each time.
unit_steps = lambda: [[PointND(t, (x, 0))] for t, x in enumerate(range(5))]

np.random.seed(0)
random_x = np.random.randn(5).cumsum()
random_x -= random_x.min()  # All x > 0
max_disp = np.diff(random_x).max()
random_walk = lambda: [[PointND(t, (x, 5))] for t, x in enumerate(random_x)]


def hash_generator(dims, box_size):
    return lambda: Hash_table(dims, box_size)


def test_search_range():
    t = link(unit_steps(), 1.1, hash_generator((10, 10), 1))
    assert len(t) == 1  # One track
    t_short = link(unit_steps(), 0.9, hash_generator((10, 10), 1))
    assert len(t_short) == len(unit_steps())  # Each step is a separate track.

    t = link(random_walk(), max_disp + 0.1, hash_generator((10, 10), 1))
    assert len(t) == 1  # One track
    t_short = link(random_walk(), max_disp - 0.1, hash_generator((10, 10), 1))
    assert len(t_short) > 1  # Multiple tracks


def test_memory():
    """A unit-stepping trajectory and a random walk are observed
    simultaneously. The random walk is missing from one observation."""
    a = [p[0] for p in unit_steps()]
    b = [p[0] for p in random_walk()]
    # b[2] is intentionally omitted below.
    gapped = lambda: deepcopy([[a[0], b[0]], [a[1], b[1]], [a[2]],
                              [a[3], b[3]], [a[4], b[4]]])
    safe_disp = 1 + random_x.max() - random_x.min()  # Definitely large enough
    t0 = link(gapped(), safe_disp, hash_generator((10, 10), 1), memory=0)
    assert len(t0) == 3, len(t0)
    t2 = link(gapped(), safe_disp, hash_generator((10, 10), 1), memory=2)
    assert len(t2) == 2, len(t2)


def test_memory_removal():
    """BUG: A particle remains in memory after its Track is resumed, leaving two
    copies that can independently pick up desinations, leaving two Points in the
    same Track in a single level."""
    levels  = []
    levels.append([PointND(0, [1, 1]), PointND(0, [4, 1])])  # two points
    levels.append([PointND(1, [1, 1])])  # one vanishes, but is remembered
    levels.append([PointND(2, [1, 1]), PointND(2, [2, 1])]) # resume Track in memory
    levels.append([PointND(3, [1, 1]), PointND(3, [2, 1]), PointND(3, [4, 1])])
    t = link(levels, 5, hash_generator((10, 10), 1), memory=2)
    assert len(t) == 3, len(t)

 
def test_memory_with_late_appearance():
    a = [p[0] for p in unit_steps()]
    b = [p[0] for p in random_walk()]
    gapped = lambda: deepcopy([[a[0]], [a[1], b[1]], [a[2]],
                              [a[3]], [a[4], b[4]]])
    safe_disp = 1 + random_x.max() - random_x.min()  # Definitely large enough
    t0 = link(gapped(), safe_disp, hash_generator((10, 10), 1), memory=1)
    assert len(t0) == 3, len(t0)
    t2 = link(gapped(), safe_disp, hash_generator((10, 10), 1), memory=4)
    assert len(t2) == 2, len(t2)


def test_box_size():
    """No matter what the box size, there should be one track, and it should
    contain all the points."""
    for box_size in [0.1, 1, 10]:
        t1 = link(unit_steps(), 1.1, hash_generator((10, 10), box_size))
        t2 = link(random_walk(), max_disp + 1,
                  hash_generator((10, 10), box_size))
        assert len(t1) == 1
        assert len(t2) == 1
        assert len(t1[0].points) == len(unit_steps())
        assert len(t2[0].points) == len(random_walk())

def test_easy_tracking():
    level_count = 5
    p_count = 16
    levels = []

    for j in range(level_count):
        level = []
        for k in np.arange(p_count) * 2:
            level.append(PointND(j, (j, k)))
        levels.append(level)

    hash_generator = lambda: Hash_table((level_count + 1,
                                            p_count * 2 + 1), .5)
    tracks = link(levels, 1.5, hash_generator)

    assert len(tracks) == p_count

    for t in tracks:
        x, y = zip(*[p.pos for p in t])
        dx = np.diff(x)
        dy = np.diff(y)

        assert np.sum(dx) == level_count - 1
        assert np.sum(dy) == 0


def test_pathological_tracking():
    level_count = 5
    p_count = 16
    levels = []
    shift = 1

    for j in range(level_count):
        level = []
        for k in np.arange(p_count) * 2:
            level.append(PointND(k // 2, (j, k + j * shift)))
        levels.append(level)

    hash_generator = lambda: Hash_table((level_count + 1,
                                            p_count*2 + level_count*shift + 1),
                                            .5)
    tracks = link(levels, 8, hash_generator)

    assert len(tracks) == p_count, len(tracks)

    for t in tracks:
        x, y = zip(*[p.pos for p in t])
        dx = np.diff(x)
        dy = np.diff(y)

