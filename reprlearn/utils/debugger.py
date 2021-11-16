from typing import Any


def print_src(pyObj: Any):
    import inspect
    lines = inspect.getsource(pyObj)
    print(lines)