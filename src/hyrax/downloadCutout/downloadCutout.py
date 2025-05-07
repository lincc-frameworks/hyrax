#!/usr/bin/env python3
# ruff: noqa: N999
import argparse
import base64
import contextlib
import csv
import dataclasses
import datetime
import errno
import getpass
import io
import logging
import math
import os
import re
import sys
import tarfile
import tempfile
import time
import urllib.request
import urllib.response
from collections.abc import Generator
from typing import IO, Any, Callable, Optional, Union, cast

import numpy as np

__all__ = []


def export(obj):
    name = obj if isinstance(obj, str) else obj.__name__
    __all__.append(name)
    return obj


api_url = "https://hsc-release.mtk.nao.ac.jp/das_cutout/pdr3"

available_reruns = [
    "pdr3_dud_rev",
    "pdr3_dud",
    "pdr3_wide",
]

default_rerun = "pdr3_wide"

available_types = [
    "coadd",
    "coadd/bg",
    "warp",
]

default_type = "coadd"

default_get_image = True
default_get_mask = False
default_get_variance = False

default_name = "{lineno}_{type}_{ra:.5f}_{dec:+.5f}_{tract}_{filter}"

default_max_connections = 4

export("ANYTRACT")
ANYTRACT = -1
export("NODIM")
NODIM = (-1, -1)
export("ALLFILTERS")
ALLFILTERS = "all"

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        fromfile_prefix_chars="@",
        description="""
            Download FITS cutouts from the website of HSC data release.
        """,
    )
    parser.add_argument(
        "--ra",
        metavar="DEGREES",
        type=parse_longitude,
        help="""
        R.A.2000.
    """,
    )
    parser.add_argument(
        "--dec",
        metavar="DEGREES",
        type=parse_latitude,
        help="""
        Dec.2000.
    """,
    )
    parser.add_argument(
        "--sw",
        metavar="DEGREES",
        type=parse_degree,
        help="""
        Semi-width in R.A. direction.
    """,
    )
    parser.add_argument(
        "--sh",
        metavar="DEGREES",
        type=parse_degree,
        help="""
        Semi-height in Dec. direction.
    """,
    )
    parser.add_argument(
        "--filter",
        type=parse_filter_opt,
        help="""
        Filter name.
    """,
    )
    parser.add_argument(
        "--rerun",
        choices=available_reruns,
        default=default_rerun,
        help="""
        Rerun name.
    """,
    )
    parser.add_argument(
        "--tract",
        type=parse_tract_opt,
        help="""
        Tract number.
    """,
    )
    parser.add_argument(
        "--image",
        metavar="BOOL",
        type=parse_bool,
        default=default_get_image,
        help=f"""
        Get the image layer. (Default: {default_get_image})
    """,
    )
    parser.add_argument(
        "--mask",
        metavar="BOOL",
        type=parse_bool,
        default=default_get_mask,
        help=f"""
        Get the mask layer. (Default: {default_get_mask})
    """,
    )
    parser.add_argument(
        "--variance",
        metavar="BOOL",
        type=parse_bool,
        default=default_get_variance,
        help=f"""
        Get the variance layer. (Default: {default_get_variance})
    """,
    )
    parser.add_argument(
        "--type",
        choices=available_types,
        default=default_type,
        help="""
        Data type.
    """,
    )
    parser.add_argument(
        "--name",
        type=str,
        default=default_name,
        help=f"""
        Output name. (python's format string; default: "{default_name}")
    """,
    )
    parser.add_argument(
        "--list",
        metavar="PATH",
        type=str,
        help="""
        Path to a coordinate list.
        If this list is given, the other command-line arguments are optional.
        Missing fields in the list will default
        to the values given in the command-line.
    """,
    )
    parser.add_argument(
        "--listtype",
        choices=["auto", "txt", "csv"],
        default="auto",
        help="""
        How to interpret the argument of --list.
        "auto" (default): Follow the extension of the file name. /
        "txt": Fields are separated by one or more spaces. /
        "csv": Comma-separated volume.
    """,
    )
    parser.add_argument(
        "--user",
        type=str,
        help="""
        User account.
    """,
    )
    parser.add_argument(
        "--password",
        type=str,
        help="""
        Password.
        If you specify --password, your password is disclosed to everybody
        on the computer you use.
        Use of --password-env is recommended instead,
        especially when this script is run on a shared-use computer.
    """,
    )
    parser.add_argument(
        "--password-env",
        metavar="ENV",
        type=str,
        default="HSC_SSP_CAS_PASSWORD",
        help="""
        Name of the environment variable from which to read password.
        Use `read -s HSC_SSP_CAS_PASSWORD` to put your password into
        $HSC_SSP_CAS_PASSWORD.
    """,
    )
    parser.add_argument(
        "--semaphore",
        metavar="PATH",
        type=str,
        default="",
        help=f"""
        Path to the named semaphore (This is not a Posix semaphore.)
        The default name is `/tmp/$(id -u -n)-downloadCutout`.
        This path must be in NFS (or any other shared filesystem)
        if you distribute the processes over a network.
        If you specify this option, `--max-connections` is {default_max_connections} by default.
    """,
    )
    parser.add_argument(
        "--max-connections",
        metavar="NUM",
        type=int,
        default=0,
        help="""
        Maximum number of connections in parallel.
        This script itself won't make parallel connections,
        but _you_ should launch this script in parallel.
        The launched processes will communicate with each other to limit
        the number of connections in parallel.
        By default `--max-connections=0`, which means limitless.
    """,
    )

    args = parser.parse_args()

    rect = Rect.create(
        rerun=args.rerun,
        type=args.type,
        filter=args.filter,
        tract=args.tract,
        ra=args.ra,
        dec=args.dec,
        sw=args.sw,
        sh=args.sh,
        image=args.image,
        mask=args.mask,
        variance=args.variance,
        name=args.name,
    )

    if not args.password:
        args.password = os.environ.get(args.password_env)

    if args.list:
        with open_inputfile(sys.stdin if args.list == "-" else args.list) as f:
            rects = read_rects(f, default=rect, type=args.listtype)
    else:
        if not rect.iscomplete():
            raise RuntimeError("Specify either (--ra --dec --sw --sh) or --list.")
        rects = [rect]

    if args.semaphore and not args.max_connections:
        args.max_connections = default_max_connections
    if args.max_connections:
        set_max_connections(args.max_connections, args.semaphore)

    download(rects, user=args.user, password=args.password, onmemory=False)


@export
@dataclasses.dataclass(order=True)
class Rect:
    """
    Rectangle to cut out of the sky.

    The constructor is not intended to be called by usual users.
    Use Rect.create() instead.

    Parameters
    ----------
    rerun
        Rerun name.
    type
        "coadd" (coadd),
        "coadd/bg" (coadd with background), or
        "warp" (warp).
    filter
        Filter name.
        This member can be `ALLFILTERS`.
    tract
        Tract number.
        This member can be `ANYTRACT`.
    ra
        R.A.2000, in degrees.
    dec
        Dec.2000, in degrees.
    sw
        Semi-width, in degrees.
    sh
        Semi-height, in degrees.
    image
        Whether to get the image layer.
    mask
        Whether to get the mask layer.
    variance
        Whether to get the variance layer.
    name
        File name format (without extension ".fits")
    lineno
        Line number in a list file.
    dim
        Dimensions of downloaded file
    """

    rerun: str = default_rerun
    type: str = default_type
    tract: int = ANYTRACT
    filter: Union[str, list[str]] = ALLFILTERS
    ra: float = math.nan
    dec: float = math.nan
    sw: float = math.nan
    sh: float = math.nan
    image: bool = default_get_image
    mask: bool = default_get_mask
    variance: bool = default_get_variance
    name: str = default_name
    lineno: int = 0
    dim: tuple[int, int] = NODIM

    @staticmethod
    def create(
        rerun: Union[str, None] = None,
        type: Union[str, None] = None,
        tract: Union[str, int, None] = None,
        filter: Union[Union[str, list[str]], None] = None,
        ra: Union[str, float, None] = None,
        dec: Union[str, float, None] = None,
        sw: Union[str, float, None] = None,
        sh: Union[str, float, None] = None,
        image: Union[str, bool, None] = None,
        mask: Union[str, bool, None] = None,
        variance: Union[str, bool, None] = None,
        name: Union[str, None] = None,
        lineno: Union[int, None] = None,
        dim: Union[tuple[int, int], None] = None,
        default: Union["Rect", None] = None,
    ) -> "Rect":
        """
        Create a Rect object.

        If any parameter is omitted,
        it defaults to the corresponding field of the `default` argument.

        Parameters
        ----------
        rerun
            Rerun name.
        type
            "coadd" (coadd),
            "coadd/bg" (coadd with background), or
            "warp" (warp).
        filter
            Filter name.
            This member can be `ALLFILTERS`.
        tract
            Tract number.
            This member can be `ANYTRACT`.
        ra
            R.A.2000, in degrees.
            This argument can be a string like "12:34:56.789" (hours),
            "12h34m56.789s", "1.2345rad" (radians), etc.
        dec
            Dec.2000, in degrees.
            This argument can be a string like "-1:23:45.678" (degrees),
            "-1d23m45.678s", "1.2345rad" (radians), etc.
        sw
            Semi-width, in degrees.
            This argument can be a string like "1.2arcmin" etc.
        sh
            Semi-height, in degrees.
            This argument can be a string like "1.2arcmin" etc.
        image
            Whether to get the image layer.
        mask
            Whether to get the mask layer.
        variance
            Whether to get the variance layer.
        name
            File name format (without extension ".fits")
        lineno
            Line number in a list file.
        dim
            Dimensions of the image in pixels.
        default
            Default value.

        Returns
        -------
        rect
            Created `Rect` object.
        """
        rect = Rect() if default is None else Rect(*dataclasses.astuple(default))

        if rerun is not None:
            rect.rerun = parse_rerun(rerun)
        if type is not None:
            rect.type = parse_type(type)
        if filter is not None:
            rect.filter = parse_filter_opt(filter)
        if tract is not None:
            rect.tract = parse_tract_opt(tract)
        if ra is not None:
            rect.ra = parse_longitude(ra)
        if dec is not None:
            rect.dec = parse_latitude(dec)
        if sw is not None:
            rect.sw = parse_degree(sw)
        if sh is not None:
            rect.sh = parse_degree(sh)
        if image is not None:
            rect.image = parse_bool(image)
        if mask is not None:
            rect.mask = parse_bool(mask)
        if variance is not None:
            rect.variance = parse_bool(variance)
        if name is not None:
            rect.name = str(name)
        if lineno is not None:
            rect.lineno = int(lineno)
        if dim is not None:
            rect.dim = dim

        return rect

    def iscomplete(self) -> bool:
        """
        Whether or not `self` is complete.

        If, for example, user creates a `Rect` object:
            rect = Rect.create(sw=25, sh=25)
        then `rect` does not have valid values of `ra` and `dec`.
        In such a case, this function returns False.

        Returns
        -------
        iscomplete
            True if `self` is complete
        """
        return self.ra == self.ra and self.dec == self.dec and self.sw == self.sw and self.sh == self.sh

    def explode(self) -> list["Rect"]:
        """
        Make copies of `self` with more specific values.

        Returns
        -------
        rects
            List of `Rect` objects, each being more specific than `self`.
        """
        if isinstance(self.filter, list):
            return [Rect.create(filter=f, default=self) for f in self.filter]
        elif self.filter == ALLFILTERS:
            return [Rect.create(filter=f, default=self) for f in _all_filters]
        else:
            return [Rect.create(default=self)]

    # Static field list used by __eq__ and __hash__
    immutable_fields = ["ra", "dec", "sw", "sh", "filter", "type", "rerun", "image", "variance", "mask"]

    def __eq__(self, obj) -> bool:
        """Define equality on Rects by sky location, size, filter, type, rerun, and image/mask/variance state.
        This allows rects can be used as keys in dictionaries while ignoring transient fields such as lineno,
        or fields that may be incorrect/changed during download process like tract or name.

        This is a compromise between
        1) Dataclass's unsafe_hash=True which would hash all fields
        and
        2) Making the dataclass frozen which would affect some of the mutability used to alter
          lineno, tract, and name

        Note that this makes equality on Rects means "the cutout API should return the same data",
        rather than "Literally all data members the same"

        Parameters
        ----------
        obj : Rect
            The rect to compare to self

        Returns
        -------
        bool
            True if the Rect's are equal
        """
        return all([self.__dict__[field] == obj.__dict__[field] for field in Rect.immutable_fields])

    def __hash__(self):
        """Define a hash function on Rects. Outside of hash collisions, this function attempts to have the
        same semantics as Rect.__eq__(). Look at Rect.__eq__() for further details.
        """
        return hash(tuple([self.__dict__[field] for field in Rect.immutable_fields]))


@export
def read_rects(
    file: Union[str, IO], default: Optional[Rect] = None, type: Optional[str] = None
) -> list[Rect]:
    """
    Read a file to get a list of `Rect` objects.

    Parameters
    ----------
    file
        A file path or a file object.
    default
        Default values.
        Fields that cannot be obtained from the file
        defaults to the corresponding fields of this object.
    type
        File type. One of "auto" (=None), "txt", "csv".
        By default, the file type is guessed
        from the extension part of the file name.

    Returns
    -------
    rects
        List of `Rect` objects.
    """
    if (not type) or type == "auto":
        isfileobj = hasattr(file, "read")
        name = str(getattr(file, "name", "(file name not available)") if isfileobj else file)
        _, ext = os.path.splitext(name)
        type = ext.lstrip(".") or "txt"

    retval: list[Rect] = []

    if type == "txt":
        retval = read_rects_from_txt(file, default=default)
    elif type == "csv":
        retval = read_rects_from_csv(file, default=default)
    else:
        raise ValueError(f"Invalid file type: {type}")

    return retval


@export
def read_rects_from_txt(file, default=None) -> list[Rect]:
    """
    Read a space-separated volume to get a list of `Rect` objects.
    The first line must contain column names.

    Parameters
    ----------
    file
        A file path or a file object.
    default
        Default values.
        Fields that cannot be obtained from the file
        defaults to the corresponding fields of this object.

    Returns
    -------
    rects
        List of `Rect` objects.
    """
    allfields = set(field.name for field in dataclasses.fields(Rect))
    rects: list[Rect] = []
    with open_inputfile(file) as f:
        f_io = io.TextIOWrapper(f, encoding="utf-8")

        fieldnames = re.sub(r"^#\??\s*", "", f_io.readline().strip().lower()).split()
        validfields = [(i, str(field)) for i, field in enumerate(fieldnames) if field in allfields]
        if not validfields:
            raise RuntimeError("No column has a valid name in the list.")

        for lineno, line in enumerate(f, start=2):
            row = line.strip().split()
            if len(row) != len(fieldnames):
                raise RuntimeError(
                    f"line {lineno}: number of fields ({len(row)}) "
                    f"does not agree with what expected ({len(fieldnames)})"
                )
            args: dict[str, Any] = {"lineno": lineno}
            args.update((field, row[i]) for i, field in validfields)
            rects.append(Rect.create(default=default, **args))

    return rects


@export
def read_rects_from_csv(file, default=None) -> list[Rect]:
    """
    Read a comma-separated volume to get a list of `Rect` objects.
    The first line must contain column names.

    Parameters
    ----------
    file
        A file path or a file object.
    default
        Default values.
        Fields that cannot be obtained from the file
        defaults to the corresponding fields of this object.

    Returns
    -------
    rects
        List of `Rect` objects.
    """
    allfields = set(field.name for field in dataclasses.fields(Rect))

    rects: list[Rect] = []
    with open_inputfile(file) as f:
        reader = csv.reader(io.TextIOWrapper(f, encoding="utf-8", newline=""))

        fieldnames = next(reader)
        if len(fieldnames) > 0:
            fieldnames[0] = re.sub(r"^#\??\s*", "", fieldnames[0].strip())
        fieldnames = [field.strip().lower() for field in fieldnames]

        validfields = [(i, str(field)) for i, field in enumerate(fieldnames) if field in allfields]
        if not validfields:
            raise RuntimeError("No column has a valid name in the list.")

        for lineno, row in enumerate(reader, start=2):
            if len(row) != len(fieldnames):
                raise RuntimeError(
                    f"line {lineno}: number of fields ({len(row)}) "
                    f"does not agree with what expected ({len(fieldnames)})"
                )
            args: dict[str, Any] = {"lineno": lineno}
            args.update((field, row[i]) for i, field in validfields)
            rects.append(Rect.create(default=default, **args))

    return rects


@contextlib.contextmanager
def open_inputfile(file: Union[str, IO]) -> Generator[IO[bytes], None, None]:
    """
    Open a file with "rb" mode.

    If `file` is a text file object, `file.buffer` will be returned.

    Parameters
    ----------
    file
        A file path or a file object.

    Returns
    -------
    contextmanager
        Context manager.
        When the context is exitted,
        the file is closed if the file has been opened by this function.
        Otherwise, the file is kept open.
    """
    if not isinstance(file, str) and hasattr(file, "read"):
        # This is already a file object
        yield getattr(file, "buffer", file)
    else:
        file = cast(str, file)
        with open(file, "rb") as f:
            yield f


def parse_rerun(s: str) -> str:
    """
    Interpret a string representing a rerun name.

    Parameters
    ----------
    s
        Rerun name.

    Returns
    -------
    rerun
        Rerun name.
    """
    lower = s.lower()
    if lower in available_reruns:
        return lower
    raise ValueError(f"Invalid rerun: {s}")


def parse_type(s: str) -> str:
    """
    Interpret a string representing an image type.

    Parameters
    ----------
    s
        Image type.

    Returns
    -------
    type
        Image type.
    """
    lower = s.lower()
    if lower in available_types:
        return lower
    raise ValueError(f"Invalid type: {s}")


def parse_tract_opt(s: Union[str, int, None]) -> int:
    """
    Interpret a string (etc) representing a tract.

    Parameters
    ----------
    s
        Tract.
        This argument may be `ANYTRACT`.

    Returns
    -------
    tract
        Tract in an integer.
    """
    if s is None:
        return ANYTRACT
    if isinstance(s, int):
        return s

    s = s.lower()
    if s == "any":
        return ANYTRACT
    return int(s)


def parse_bool(s: Union[str, bool]) -> bool:
    """
    Interpret a string (etc) representing a boolean value.

    Parameters
    ----------
    s
        A string (etc) representing a boolean value.

    Returns
    -------
    b
        True or False.
    """
    if isinstance(s, bool):
        return s

    if isinstance(s, np.bool):
        return s

    return {
        "false": False,
        "f": False,
        "no": False,
        "n": False,
        "off": False,
        "0": False,
        "true": True,
        "t": True,
        "yes": True,
        "y": True,
        "on": True,
        "1": True,
    }[s.lower()]


def parse_longitude(s: Union[str, float]) -> float:
    """
    Interpret a longitude.

    Parameters
    ----------
    s
        A string representing a longitude,
        or a float value in degrees.

    Returns
    -------
    longitude
        Degrees.
    """
    type, value = _parse_angle(s)
    if type == "sex":
        return 15 * value
    else:
        return value


def parse_latitude(s: Union[str, float]) -> float:
    """
    Interpret a latitude.

    Parameters
    ----------
    s
        A string representing a latitude,
        or a float value in degrees.

    Returns
    -------
    latitude
        Degrees.
    """
    type, value = _parse_angle(s)
    return value


def parse_degree(s: Union[str, float]) -> float:
    """
    Interpret an angle, which is in degrees by default.

    Parameters
    ----------
    s
        A string representing an angle,
        or a float value in degrees.

    Returns
    -------
    angle
        Degrees.
    """
    type, value = _parse_angle(s)
    return value


def _parse_angle(s: Union[str, float]) -> tuple[str, float]:
    """
    Interpret an angle.

    Parameters
    ----------
    s
        A string representing an angle.

    Returns
    -------
    type
      - "bare"
        `s` did not have its unit.
        What `angle` means must be decided by the caller.
      - "sex"
        `s` was in "99:99:99.999". It may be hours or degrees.
        What `angle` means must be decided by the caller.
      - "deg"
        `angle` is in degrees.

    angle
        a float value
    """
    try:
        if isinstance(s, (float, int)):
            return "bare", float(s)

        s = re.sub(r"\s", "", s).lower()
        m = re.match(
            r"\A(.+)(deg|degrees?|amin|arcmin|arcminutes?|asec|arcsec|arcseconds?|rad|radians?)\Z", s
        )
        if m:
            value, unit = m.groups()
            return "deg", float(value) * _angle_units[unit]

        m = re.match(r"\A([+\-]?)([0-9].*)d([0-9].*)m([0-9].*)s\Z", s)
        if m:
            sign_s, degrees, minutes, seconds = m.groups()
            sign = -1.0 if sign_s == "-" else 1.0
            return "deg", sign * (float(seconds) / 3600 + float(minutes) / 60 + float(degrees))

        m = re.match(r"\A([+\-]?)([0-9].*)h([0-9].*)m([0-9].*)s\Z", s)
        if m:
            sign_s, hours, minutes, seconds = m.groups()
            sign = -1.0 if sign_s == "-" else 1.0
            return "deg", 15.0 * sign * (float(seconds) / 3600 + float(minutes) / 60 + float(hours))

        m = re.match(r"\A([+\-]?)([0-9].*):([0-9].*):([0-9].*)\Z", s)
        if m:
            sign_s, degrees, minutes, seconds = m.groups()
            sign = -1.0 if sign_s == "-" else 1.0
            return "sex", sign * (float(seconds) / 3600 + float(minutes) / 60 + float(degrees))

        return "bare", float(s)

    except Exception:
        raise ValueError(f"Cannot interpret angle: '{s}'") from None


_angle_units = {
    "deg": 1.0,
    "degree": 1.0,
    "degrees": 1.0,
    "amin": 1.0 / 60,
    "arcmin": 1.0 / 60,
    "arcminute": 1.0 / 60,
    "arcminutes": 1.0 / 60,
    "asec": 1.0 / 3600,
    "arcsec": 1.0 / 3600,
    "arcsecond": 1.0 / 3600,
    "arcseconds": 1.0 / 3600,
    "rad": 180 / math.pi,
    "radian": 180 / math.pi,
    "radians": 180 / math.pi,
}


def parse_filter(s: str) -> str:
    """
    Interpret a filter name.

    Parameters
    ----------
    s
        A string representing a filter.
        This may be an alias of a filter name.
        (Like "g" for "HSC-G")

    Returns
    -------
    filter
        A filter name.
    """
    if s in _all_filters:
        return s

    for physicalname, info in _all_filters.items():
        if s in info["alias"]:
            return physicalname

    raise ValueError(f"filter '{s}' not found.")


_all_filters = dict(
    [
        ("HSC-G", {"alias": {"W-S-G+", "g"}, "display": "g"}),
        ("HSC-R", {"alias": {"W-S-R+", "r"}, "display": "r"}),
        ("HSC-I", {"alias": {"W-S-I+", "i"}, "display": "i"}),
        ("HSC-Z", {"alias": {"W-S-Z+", "z"}, "display": "z"}),
        ("HSC-Y", {"alias": {"W-S-ZR", "y"}, "display": "y"}),
        ("IB0945", {"alias": {"I945"}, "display": "I945"}),
        ("NB0387", {"alias": {"N387"}, "display": "N387"}),
        ("NB0400", {"alias": {"N400"}, "display": "N400"}),
        ("NB0468", {"alias": {"N468"}, "display": "N468"}),
        ("NB0515", {"alias": {"N515"}, "display": "N515"}),
        ("NB0527", {"alias": {"N527"}, "display": "N527"}),
        ("NB0656", {"alias": {"N656"}, "display": "N656"}),
        ("NB0718", {"alias": {"N718"}, "display": "N718"}),
        ("NB0816", {"alias": {"N816"}, "display": "N816"}),
        ("NB0921", {"alias": {"N921"}, "display": "N921"}),
        ("NB0926", {"alias": {"N926"}, "display": "N926"}),
        ("NB0973", {"alias": {"N973"}, "display": "N973"}),
        ("NB1010", {"alias": {"N1010"}, "display": "N1010"}),
        ("ENG-R1", {"alias": {"109", "r1"}, "display": "r1"}),
        ("PH", {"alias": {"PH"}, "display": "PH"}),
        ("SH", {"alias": {"SH"}, "display": "SH"}),
        ("MegaCam-u", {"alias": {"u2"}, "display": "MegaCam-u"}),
        ("MegaCam-uS", {"alias": {"u1"}, "display": "MegaCam-uS"}),
        ("VIRCAM-H", {"alias": {"Hvir", "hvir"}, "display": "VIRCAM-H"}),
        ("VIRCAM-J", {"alias": {"Jvir", "jvir"}, "display": "VIRCAM-J"}),
        ("VIRCAM-Ks", {"alias": {"Ksvir", "ksvir"}, "display": "VIRCAM-Ks"}),
        ("VIRCAM-NB118", {"alias": {"NB118vir", "n118vir"}, "display": "VIRCAM-NB118"}),
        ("VIRCAM-Y", {"alias": {"Yvir", "yvir"}, "display": "VIRCAM-Y"}),
        ("WFCAM-H", {"alias": {"Hwf", "hwf"}, "display": "WFCAM-H"}),
        ("WFCAM-J", {"alias": {"Jwf", "jwf"}, "display": "WFCAM-J"}),
        ("WFCAM-K", {"alias": {"Kwf", "kwf"}, "display": "WFCAM-K"}),
    ]
)


def parse_filter_opt(s: Optional[Union[str, list]]) -> Union[str, list]:
    """
    Interpret a filter name.
    The argument may be `ALLFILTERS`.or None
    (both have the same meaning).

    Parameters
    ----------
    s
        A string representing a filter.
        This may be an alias of a filter name.
        (Like "g" for "HSC-G")
        Or it may be `ALLFILTERS`.
        If `s` is None, it has the same meaning as `ALLFILTERS`.

    Returns
    -------
    filter
        A filter name.
        This can be `ALLFILTERS`
    """
    if s is None:
        return ALLFILTERS
    if isinstance(s, list):
        return [parse_filter(filter) for filter in s]
    if s.lower() == ALLFILTERS:
        return ALLFILTERS
    return parse_filter(s)


@export
def download(
    rects: Union[Rect, list[Rect]],
    user: Optional[str] = None,
    password: Optional[str] = None,
    *,
    onmemory: bool = True,
    **kwargs__download,
) -> Union[list, list[list], None]:
    """
    Cut `rects` out of the sky.

    Parameters
    ----------
    rects : Union[Rect, list[Rect]]
        A `Rect` object or a list of `Rect` objects to download
    user : Optional[str]
        Username. If None, it will be asked interactively.
    password : Optional [str]
        Password. If None, it will be asked interactively.
    onmemory, optional : bool
        Return `datalist` on memory.
        If `onmemory` is False, downloaded cut-outs are written to files.
    kwargs__download: dict
        Additional keyword args are passed through to _download and eventually to _download_chunk() and
        urllib.request.urlopen.

        Some important (but entirely optional!) keyword args processed later in the download callstack are
        listed below. Anything urllib.request.urlopen will accept is fair game too!

            chunksize : int
                The number of rects to include in a single http request. Default 990 rects. See _download()
                for greater detail.
            retries : int
                The number of times to retry a failed http request before moving on or attempting to trigger
                failed_chunk_handler. Default is 3. See _download() for greater detail.
            retrywait : int
                Base seconds to wait between retries. Acts as base for exponential retry formula of
                base * (2**attempt_num) seconds. Default is 30 s. See _download() for greater detail.
            failed_chunk_hook : Callable[[list[Rect], Exception, int], Any]
                Hook which is called every time a chunk fails `retries` time. The arguments to the hook are
                the rects in the failed chunk, the exception encountered while making the last request, and
                the number of attempts. See _download() for greater detail.
            request_hook : Callable[[urllib.request.Request, datetime.datetime, datetime.datetime, int, int]
                Optional hook called on every request. See _download_chunk() for greater detail

    Returns
    -------
    datalist : Union[list, list[list], None]
        If onmemory == False, `datalist` is None.
        If onmemory == True:
          - If `rects` is a simple `Rect` object,
            `datalist[j]` is a tuple `(metadata: dict, data: bytes)`.
            This is a list because there may be more than one file
            for a single `Rect` (if, say, filter==ALLFILTERS).
            This list may also be empty, which means no data was found.
          - If `rects` is a list of `Rect` objects,
            `datalist[i]` corresponds to `rects[i]`, and
            `datalist[i][j]` is a tuple `(metadata: dict, data: bytes)`.
    """
    isscalar = isinstance(rects, Rect)
    if isscalar:
        rects = [cast(Rect, rects)]
    rects = cast(list[Rect], rects)

    ret = _download(rects, user, password, onmemory=onmemory, **kwargs__download)
    if isscalar and onmemory:
        ret = cast(list[list], ret)
        return ret[0]

    return ret


def _download(
    rects: list[Rect],
    user: Optional[str],
    password: Optional[str],
    *,
    onmemory: bool,
    chunksize: int = 990,
    retries: int = 3,
    retrywait: int = 30,
    **kwargs__download_chunk,
) -> Optional[list[list]]:
    """
    Cut `rects` out of the sky. Implements configurable request size, retries and exponential backoff.

    When `onmemory == False` this function saves snapshots to the current working directory.
    onmemory has no default and must be specified.

    When `onmemory == False ` and the current working directory contains resume data, and
    `resume = True` (default: False), then the download will be resumed.

    Parameters
    ----------
    rects
        A list of `Rect` objects
    user
        Username. If None, it will be asked interactively.
    password
        Password. If None, it will be asked interactively.
    onmemory: bool
        Return `datalist` on memory.
        If `onmemory` is False, downloaded cut-outs are written to files in the current working directory.
    chunksize: int, optional
        Number of cutout lines to pack into a single request. Defaults to 990 if unspecified.
    retries: int, optional
        Number of attempts to make to fetch each chunk. Defaults to 3 if unspecified.
    retrywait: int, optional
        Base number of seconds to wait between retries. Retry waits are computed using an exponential backoff
        where the retry time for attempts is calculated as retrywait * (2 ** attempt) seconds , with attempt=0
        for the first wait.
    kwargs__download_chunk: dict, optional
        Additional keyword args are passed through to _download_chunk()

    Returns
    -------
    datalist
        If onmemory == False, `datalist` is None.
        If onmemory == True,
        `datalist[i]` corresponds to `rects[i]`, and
        `datalist[i][j]` is a tuple `(metadata: dict, data: bytes)`.
    """
    if not rects:
        return [] if onmemory else None

    for rect in rects:
        if not rect.iscomplete():
            raise RuntimeError(f"'ra', 'dec', 'sw', and 'sh' must be specified: {rect}")

    exploded_rects: list[tuple[Rect, int]] = []
    for index, rect in enumerate(rects):
        exploded_rects.extend((r, index) for r in rect.explode())

    # Sort the rects so that the server can use cache
    # as frequently as possible.
    # We will later use `index` to sort them back.
    exploded_rects.sort()

    if not user:
        user = input("username? ").strip()
        if not user:
            raise RuntimeError("User name is empty.")

    if not password:
        password = getpass.getpass(prompt="password? ")
        if not password:
            raise RuntimeError("Password is empty.")

    datalist: list[tuple[int, dict, bytes]] = []

    # Chunk loop
    for i in range(0, len(exploded_rects), chunksize):
        # Retry loop
        for attempt in range(0, retries):
            try:
                ret = _download_chunk(
                    exploded_rects[i : i + chunksize],
                    user,
                    password,
                    onmemory=onmemory,
                    **kwargs__download_chunk,
                )
                break
            except KeyboardInterrupt:
                logger.critical("Keyboard Interrupt recieved.")
                raise
            except Exception as exception:
                # Humans count attempts from 1, this loop counts from zero.
                logger.warning(
                    f"Attempt {attempt + 1} of {retries} to request rects [{i}:{i + chunksize}] has error:"
                )
                logger.warning(exception)

                # If the final attempt on this chunk fails, we move on.
                if attempt + 1 == retries:
                    break
                # Otherwise wait for exponential backoff and try again
                else:
                    backoff = retrywait * (2**attempt)
                    if backoff != 0:
                        logger.info(f"Retrying in {backoff} seconds... ")
                        time.sleep(backoff)
                        logger.info("Retrying now")
                    continue
        if onmemory:
            datalist += cast(list, ret)

    if onmemory:
        returnedlist: list[list[tuple[dict, bytes]]] = [[] for i in range(len(rects))]
        for index, metadata, data in datalist:
            returnedlist[index].append((metadata, data))

    return returnedlist if onmemory else None


def _download_chunk(
    rects: list[tuple[Rect, Any]],
    user: str,
    password: str,
    manifest: Optional[dict[Rect, str]],
    *,
    onmemory: bool,
    request_hook: Optional[
        Callable[[urllib.request.Request, datetime.datetime, datetime.datetime, int, int], Any]
    ] = None,
    rect_hook: Optional[Callable[[Rect, str], Any]] = None,
    **kwargs_urlopen,
) -> Optional[list]:
    """
    Cut `rects` out of the sky.

    Parameters
    ----------
    rects
        A list of `(Rect, Any)`.
        The length of this list must be smaller than the server's limit.
        Each `Rect` object must be explode()ed beforehand.
        The `Any` value attached to each `Rect` object is a marker.
        The marker is used to indicate the `Rect` in the returned list.
    user
        Username.
    password
        Password.
    manifest
        Dictionary from Rect to filename. If Provided, this function will fill in as it downloads.
        If download of a file fails, the file's entry will read "Attempted" for the filename.
    onmemory
        Return `datalist` on memory.
        If `onmemory` is False, downloaded cut-outs are written to files.
    request_hook
        Function that is called with the response of all requests made
        Intended to support bandwidth instrumentation.
    rect_hook
        Function to be called on every rectangle downloaded. The callback recieves the rect and the filename
        as arguments
    kwargs_urlopen
        Additional keyword args are passed through to urllib.request.urlopen

    Returns
    -------
    datalist
        If onmemory == False, `datalist` is None.
        If onmemory == True,
        each element is a tuple `(marker: Any, metadata: dict, data: bytes)`.
        For `marker`, see the comment for the parameter `rects`.
        Two or more elements in this list may result
        from a single `Rect` object.
    """
    fields = list(_format_rect_member.keys())
    coordlist = [f"#? {' '.join(fields)}"]
    for rect, _ in rects:
        coordlist.append(" ".join(_format_rect_member[field](getattr(rect, field)) for field in fields))

    boundary = "Boundary"
    header = f'--{boundary}\r\nContent-Disposition: form-data; name="list"; filename="coordlist.txt"\r\n\r\n'
    footer = f"\r\n--{boundary}--\r\n"

    data = (header + "\n".join(coordlist) + footer).encode("utf-8")
    secret = base64.standard_b64encode(f"{user}:{password}".encode()).decode("ascii")

    req = urllib.request.Request(
        api_url.rstrip("/") + "/cgi-bin/cutout",
        data=data,
        headers={
            "Authorization": f"Basic {secret}",
            "Content-Type": f'multipart/form-data; boundary="{boundary}"',
        },
        method="POST",
    )

    returnedlist = []

    # Set timeout to 1 hour if no timout was set higher up
    kwargs_urlopen.setdefault("timeout", 3600)

    # Set all manifest entries to indicate an attempt was made.
    if manifest is not None:
        for rect, _ in rects:
            manifest[rect] = "Attempted"

    with get_connection_semaphore():
        request_started = datetime.datetime.now()
        with urllib.request.urlopen(req, **kwargs_urlopen) as fin:
            response_started = datetime.datetime.now()
            response_size = 0
            with tarfile.open(fileobj=fin, mode="r|") as tar:
                for info in tar:
                    fitem = tar.extractfile(info)
                    response_size += info.size
                    if fitem is None:
                        continue
                    with fitem:
                        metadata = _tar_decompose_item_name(info.name)
                        rect, index = rects[metadata["lineno"] - 2]
                        # Overwrite metadata's lineno (= lineno in this chunk)
                        # with rect's lineno (= global lineno)
                        # for fear of confusion.
                        metadata["lineno"] = rect.lineno
                        # Overwrite rect's tract (which may be ANYTRACT)
                        # with metadata's tract (which is always a valid value)
                        # for fear of confusion.
                        rect.tract = metadata["tract"]
                        metadata["rect"] = rect
                        if onmemory:
                            returnedlist.append((index, metadata, fitem.read()))
                        else:
                            filename = make_filename(metadata)
                            dirname = os.path.dirname(filename)
                            if dirname:
                                os.makedirs(dirname, exist_ok=True)
                            with open(filename, "wb") as fout:
                                _splice(fitem, fout)
                            if rect_hook:
                                rect_hook(rect, filename)
                            if manifest is not None:
                                manifest[rect] = filename
            if request_hook:
                request_hook(req, request_started, response_started, response_size, len(rects))

    return returnedlist if onmemory else None


_format_rect_member: dict[str, Callable[[str], Any]] = {
    "rerun": str,
    "type": str,
    "filter": str,
    "tract": lambda x: ("any" if x == ANYTRACT else str(x)),
    "ra": lambda x: f"{x:.16e}deg",
    "dec": lambda x: f"{x:.16e}deg",
    "sw": lambda x: f"{x:.16e}deg",
    "sh": lambda x: f"{x:.16e}deg",
    "image": lambda x: ("true" if x else "false"),
    "mask": lambda x: ("true" if x else "false"),
    "variance": lambda x: ("true" if x else "false"),
}


def _tar_decompose_item_name(name: str) -> dict:
    """
    Get a metadata dictionary by decomposing an item name in a tar file.

    Parameters
    ----------
    name
        The name of an item in a tar file returned by the server.

    Returns
    -------
    metadata
        A dictionary that has the following keys:
          - "lineno": Line number (starting with 2).
          - "type": "coadd", "coadd/bg", or "warp".
          - "filter": Filter name.
          - "tract": Tract number.
          - "rerun": Rerun name.
          - "visit": (warp only) Visit number.
    """
    m = re.fullmatch(
        r"arch-[0-9]+-[0-9]+/(?P<lineno>[0-9]+)-(?P<type>cutout|coadd\+bg)-(?P<filter>[^/]+)-(?P<tract>[0-9]+)-(?P<rerun>[^/]+)\.fits",
        name,
    )
    if m:
        metadata: dict[str, Any] = m.groupdict()
        metadata["lineno"] = int(metadata["lineno"])
        metadata["type"] = {"cutout": "coadd", "coadd+bg": "coadd/bg"}[metadata["type"]]
        metadata["tract"] = int(metadata["tract"])
        return metadata

    m = re.fullmatch(
        r"arch-[0-9]+-[0-9]+/(?P<lineno>[0-9]+)-warps-(?P<filter>[^/]+)-(?P<tract>[0-9]+)-(?P<rerun>[^/]+)/warp-(?P<visit>[0-9]+)\.fits",
        name,
    )
    if m:
        metadata: dict[str, Any] = m.groupdict()  # type: ignore[no-redef]
        metadata["lineno"] = int(metadata["lineno"])
        metadata["type"] = "warp"
        metadata["tract"] = int(metadata["tract"])
        metadata["visit"] = int(metadata["visit"])
        return metadata

    raise ValueError("File name not interpretable")


@export
def make_filename(metadata: dict) -> str:
    """
    Make a filename from `metadata` that is returned by `download(onmemory=True)`.

    Parameters
    ----------
    metadata
        A metadata dictionary.

    Returns
    -------
    filename
        File name.
    """
    rect = metadata["rect"]
    args = dataclasses.asdict(rect)
    type = args["type"]
    if type == "warp":
        args["name"] = f"{args['name']}_{metadata['visit']:06d}"
    if type == "coadd/bg":
        args["type"] = "coadd+bg"

    name = args.pop("name")
    return str(name.format(**args)) + ".fits"


def _splice(fin: IO[bytes], fout: IO[bytes]):
    """
    Read from `fin` and write to `fout` until the end of file.

    Parameters
    ----------
    fin
        Input file.
    fout
        Output file.
    """
    buffer = memoryview(bytearray(10485760))
    while True:
        n = fin.readinto(buffer)  # type: ignore[attr-defined]
        if n <= 0:
            break
        fout.write(buffer[:n])


class Semaphore:
    """
    Named semaphore.

    This semaphore can be shared by multiple machines
    that mount a shared NFS.

    It is guaranteed that the semaphore locked by this process
    is automatically unlocked as soon as this process terminates.

    Parameters
    ----------
    init_num
        Initial semaphore count.
        Zero or negative values mean infinity.
    path
        Path of the semaphore.
        E.g. "/tmp/semaphore-cutout", "/share/data/sem_cutout" etc.
        Be warned that many files will be made under this path (directory).
    """

    def __init__(self, init_num: int, path: str):
        self.init_num = init_num
        self.path = path
        self.fd = -1

        if init_num <= 0:
            return

        os.makedirs(path, exist_ok=True)

    def __del__(self):
        self.unlock()

    def __enter__(self) -> "Semaphore":
        self.lock()
        return self

    def __exit__(self, *args):
        self.unlock()

    def lock(self):
        """
        Wait the semaphore.
        """
        if self.init_num <= 0:
            return
        if self.fd != -1:
            return

        fd = -1

        try:
            while True:
                for i in range(self.init_num):
                    fd = os.open(os.path.join(self.path, f"{i}.sem"), os.O_RDWR | os.O_CREAT, 0o644)
                    try:
                        os.lockf(fd, os.F_TLOCK, 0)
                        self.fd = fd
                        fd = -1
                        return
                    except OSError as e:
                        if e.errno not in (errno.EACCES, errno.EAGAIN):
                            raise

                    os.close(fd)
                    fd = -1

                time.sleep(0.01 * self.init_num)
        finally:
            if fd != -1:
                os.close(fd)

    def unlock(self):
        """
        Signal the semaphore.
        """
        if self.init_num <= 0:
            return
        if self.fd != -1:
            tempfd = self.fd
            self.fd = -1
            os.close(tempfd)


_sem_connections = Semaphore(0, "")


@export
def set_max_connections(num: int, semaphore_path: str = ""):
    """
    Set maximum number of connections.

    Parameters
    ----------
    num
        Number of connections. (Limitless if zero or negative)
    semaphore_path
        Path to the semaphore.
        The default name is `/tmp/$(id -u -n)-downloadCutout`.
    """
    global _sem_connections

    if num > 0 and not semaphore_path:
        semaphore_path = os.path.join(tempfile.gettempdir(), f"{getpass.getuser()}-downloadCutout")

    _sem_connections = Semaphore(num, semaphore_path)


def get_connection_semaphore() -> Semaphore:
    """
    Get the semaphore configured by `set_max_connections()`

    Returns
    -------
    semaphore
        Semaphore configured by `set_max_connections()`
    """
    return _sem_connections


if __name__ == "__main__":
    main()
