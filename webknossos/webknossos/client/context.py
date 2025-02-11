"""
# Authentication & Server Context

When interacting with a webKnossos server, you might need to
specify your user token to authenticate yourself.
You can copy your token from

[https://webknossos.org/auth/token](https://webknossos.org/auth/token)

Using the same methods, you can also specify the webknossos-server if you
are not using the default [webknossos.org](https://webknossos.org) instance.

There are the following four options to specify which server context to use:

1. Specifying the context in code using the `webknossos_context`
   contextmanager in a `with` statement:

    ```python
    with webknossos_context(token="my_webknossos_token"):
       # code that interacts with webknossos
    ```

    For more information about the `with` statement and contextmanagers,
    please see [this tutorial](https://realpython.com/python-with-statement).

2. You may specify your settings as environment variables `WK_TOKEN` and `WK_URL`:

    ```shell
    WK_TOKEN="my_webknossos_token" python my-script.py
    ```
3. You can also specify those environment variables in a `.env` file
   in your working directory.
   Environment variables set in the command line take precedence.

    ```shell
    # content of .env
    WK_TOKEN="my_webknossos_token"
    WK_URL="…"
    ```

4. If nothing else is specified and authentication is needed,
   you are asked interactively for a token, which is used for
   subsequent interactions in the same python run as well.
"""


import os
from contextlib import contextmanager
from contextvars import ContextVar
from functools import lru_cache
from typing import Iterator, Optional

import attr
from dotenv import load_dotenv
from rich.prompt import Prompt

from webknossos.client._defaults import DEFAULT_WEBKNOSSOS_URL
from webknossos.client._generated import Client as GeneratedClient

load_dotenv()


@lru_cache(maxsize=None)
def _cached_ask_for_token(webknossos_url: str) -> str:
    # TODO  pylint: disable=fixme
    # -validate token and ask again if necessary
    # -ask if the token should be saved in some .env file
    # -reset invalid tokens
    #  (e.g. use cachetools for explicit cache managment:
    #  https://cachetools.readthedocs.io/en/stable/#memoizing-decorators)
    return Prompt.ask(
        f"\nPlease enter your webknossos token as shown on {webknossos_url}/auth/token ",
        password=True,
    )


@lru_cache(maxsize=None)
def _cached_get_org(context: "_WebknossosContext") -> str:
    from webknossos.client._generated.api.default import current_user_info

    current_user_info_response = current_user_info.sync(
        client=context.generated_auth_client
    )
    assert current_user_info_response is not None
    return current_user_info_response.organization


# TODO reset invalid tokens e.g. using cachetools  pylint: disable=fixme
@lru_cache(maxsize=None)
def _cached_get_datastore_token(context: "_WebknossosContext") -> str:
    from webknossos.client._generated.api.default import generate_token_for_data_store

    generate_token_for_data_store_response = generate_token_for_data_store.sync(
        client=context.generated_auth_client
    )
    assert generate_token_for_data_store_response is not None
    return generate_token_for_data_store_response.token


@lru_cache(maxsize=None)
def _cached__get_generated_client(
    webknossos_url: str,
    token: Optional[str],
) -> GeneratedClient:
    """Generates a client which might contain an x-auth-token header."""
    if token is None:
        return GeneratedClient(base_url=webknossos_url)
    else:
        return GeneratedClient(
            base_url=webknossos_url,
            headers={"X-Auth-Token": token},
        )


@attr.frozen
class _WebknossosContext:
    url: str = os.environ.get("WK_URL", default=DEFAULT_WEBKNOSSOS_URL)
    token: Optional[str] = os.environ.get("WK_TOKEN", default=None)

    # all properties are cached outside to allow re-usability
    # if same context is instantiated twice
    @property
    def required_token(self) -> str:
        if self.token is None:
            return _cached_ask_for_token(self.url)
        else:
            return self.token

    @property
    def organization(self) -> str:
        return _cached_get_org(self)

    @property
    def datastore_token(self) -> str:
        return _cached_get_datastore_token(self)

    @property
    def generated_client(self) -> GeneratedClient:
        return _cached__get_generated_client(self.url, self.token)

    @property
    def generated_auth_client(self) -> GeneratedClient:
        return _cached__get_generated_client(self.url, self.required_token)


_webknossos_context_var: ContextVar[_WebknossosContext] = ContextVar(
    "_webknossos_context_var", default=_WebknossosContext()
)


@contextmanager
def webknossos_context(
    url: str = DEFAULT_WEBKNOSSOS_URL,
    token: Optional[str] = None,
) -> Iterator[None]:
    context_var_token = _webknossos_context_var.set(_WebknossosContext(url, token))
    try:
        yield
    finally:
        _webknossos_context_var.reset(context_var_token)


def _get_context() -> _WebknossosContext:
    return _webknossos_context_var.get()


def _get_generated_client(enforce_auth: bool = False) -> GeneratedClient:
    if enforce_auth:
        return _get_context().generated_auth_client
    else:
        return _get_context().generated_client
