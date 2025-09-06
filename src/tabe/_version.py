from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version('tabe')
except PackageNotFoundError:
    __version__ = 'dev'

__all__ = ['__version__']
