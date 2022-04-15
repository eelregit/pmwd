from importlib.metadata import version, PackageNotFoundError


try:
    __version__ = version('pmwd')
except PackageNotFoundError:
    pass  # not installed
