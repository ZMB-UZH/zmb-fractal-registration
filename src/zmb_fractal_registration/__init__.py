"""Package description."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("zmb_fractal_registration")
except PackageNotFoundError:
    __version__ = "uninstalled"
