import pytest

from webknossos.client.context import _get_context, _WebknossosContext


@pytest.fixture
def env_context() -> _WebknossosContext:
    return _get_context()


# pylint: disable=redefined-outer-name


@pytest.mark.vcr()
def test_user_organization(env_context: _WebknossosContext) -> None:
    assert env_context.organization == "scalable_minds"
