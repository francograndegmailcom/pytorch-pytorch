from contextlib import contextmanager

try:
    from torch._C import _itt
except ImportError:
    class _ITTStub(object):
        @staticmethod
        def _fail(*args, **kwargs):
            raise RuntimeError("ITT functions not installed. Are you sure you have a ITT build?")

        rangePush = _fail
        rangePop = _fail
        mark = _fail

    _itt = _ITTStub()  # type: ignore[assignment]


__all__ = ['range_push', 'range_pop', 'mark', 'range']


def range_push(msg):
    """
    Pushes a range onto a stack of nested range span.  Returns zero-based
    depth of the range that is started.

    Arguments:
        msg (str): ASCII message to associate with range
    """
    return _itt.rangePush(msg)


def range_pop():
    """
    Pops a range off of a stack of nested range spans. Returns the
    zero-based depth of the range that is ended.
    """
    return _itt.rangePop()


def mark(msg):
    """
    Describe an instantaneous event that occurred at some point.

    Arguments:
        msg (str): ASCII message to associate with the event.
    """
    return _itt.mark(msg)


@contextmanager
def range(msg, *args, **kwargs):
    """
    Context manager / decorator that pushes an ITT range at the beginning
    of its scope, and pops it at the end. If extra arguments are given,
    they are passed as arguments to msg.format().

    Args:
        msg (str): message to associate with the range
    """
    range_push(msg.format(*args, **kwargs))
    yield
    range_pop()
