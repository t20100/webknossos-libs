import numpy as np
import pytest

from webknossos.geometry import BoundingBox, Mag


def test_align_with_mag_ceiled() -> None:

    assert BoundingBox((1, 1, 1), (10, 10, 10)).align_with_mag(
        Mag(2), ceil=True
    ) == BoundingBox(topleft=(0, 0, 0), size=(12, 12, 12))
    assert BoundingBox((1, 1, 1), (9, 9, 9)).align_with_mag(
        Mag(2), ceil=True
    ) == BoundingBox(topleft=(0, 0, 0), size=(10, 10, 10))
    assert BoundingBox((1, 1, 1), (9, 9, 9)).align_with_mag(
        Mag(4), ceil=True
    ) == BoundingBox(topleft=(0, 0, 0), size=(12, 12, 12))
    assert BoundingBox((1, 2, 3), (9, 9, 9)).align_with_mag(
        Mag(2), ceil=True
    ) == BoundingBox(topleft=(0, 2, 2), size=(10, 10, 10))


def test_align_with_mag_floored() -> None:

    assert BoundingBox((1, 1, 1), (10, 10, 10)).align_with_mag(Mag(2)) == BoundingBox(
        topleft=(2, 2, 2), size=(8, 8, 8)
    )
    assert BoundingBox((1, 1, 1), (9, 9, 9)).align_with_mag(Mag(2)) == BoundingBox(
        topleft=(2, 2, 2), size=(8, 8, 8)
    )
    assert BoundingBox((1, 1, 1), (9, 9, 9)).align_with_mag(Mag(4)) == BoundingBox(
        topleft=(4, 4, 4), size=(4, 4, 4)
    )
    assert BoundingBox((1, 2, 3), (9, 9, 9)).align_with_mag(Mag(2)) == BoundingBox(
        topleft=(2, 2, 4), size=(8, 8, 8)
    )


def test_in_mag() -> None:

    with pytest.raises(AssertionError):
        BoundingBox((1, 2, 3), (9, 9, 9)).in_mag(Mag(2))

    with pytest.raises(AssertionError):
        BoundingBox((2, 2, 2), (9, 9, 9)).in_mag(Mag(2))

    assert BoundingBox((2, 2, 2), (10, 10, 10)).in_mag(Mag(2)) == BoundingBox(
        topleft=(1, 1, 1), size=(5, 5, 5)
    )


def test_with_bounds() -> None:

    assert BoundingBox((1, 2, 3), (5, 5, 5)).with_bounds_x(0, 10) == BoundingBox(
        (0, 2, 3), (10, 5, 5)
    )
    assert BoundingBox((1, 2, 3), (5, 5, 5)).with_bounds_y(
        new_topleft_y=0
    ) == BoundingBox((1, 0, 3), (5, 5, 5))
    assert BoundingBox((1, 2, 3), (5, 5, 5)).with_bounds_z(
        new_size_z=10
    ) == BoundingBox((1, 2, 3), (5, 5, 10))


def test_contains() -> None:

    assert BoundingBox((1, 1, 1), (5, 5, 5)).contains((2, 2, 2))
    assert BoundingBox((1, 1, 1), (5, 5, 5)).contains((1, 1, 1))

    # top-left is inclusive, bottom-right is exclusive
    assert not BoundingBox((1, 1, 1), (5, 5, 5)).contains((6, 6, 6))
    assert not BoundingBox((1, 1, 1), (5, 5, 5)).contains((20, 20, 20))

    # nd-array may contain float values
    assert BoundingBox((1, 1, 1), (5, 5, 5)).contains(np.array([5.5, 5.5, 5.5]))
    assert not BoundingBox((1, 1, 1), (5, 5, 5)).contains(np.array([6.0, 6.0, 6.0]))
