import pytest

from webknossos import User


def assert_valid_user(user: User) -> None:
    assert user.id
    assert user.first_name
    assert user.last_name
    assert user.email
    assert len(user.teams) > 0


@pytest.mark.vcr()
def test_get_current_user() -> None:
    assert_valid_user(User.get_current_user())


@pytest.mark.vcr()
def test_get_logged_time() -> None:
    logged_time = User.get_current_user().get_logged_times()
    assert len(logged_time) > 0
    assert sum(i.duration_in_seconds for i in logged_time) > 0


@pytest.mark.vcr()
def test_get_all_managed_users() -> None:
    users = User.get_all_managed_users()
    assert len(users) > 0
    for user in users:
        assert_valid_user(user)
