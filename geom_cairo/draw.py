from geom.data.rect import Rect
from geom.data.circle import Circle
from geom.data.ellipse import Ellipse
from geom.data.polygon import Polygon
from geom.data.polyline import Polyline
from geom.data.point import Point
from geom.data.line import Line
from geom.data.cubic import Cubic
from geom.data.arc import Arc

import numpy as np
import cairo

from .cairo.ctx import (
    _setup,
    clear_canvas,
    set_range,
    set_bounds,
    rect,
    circle,
    arc,
    ellipse,
    path,
    line,
    point,
    curve,
    surface_out,
    surface_to_numpy_array,
    paint_surface,
    get_canvas_size,
    get_canvas_bounds,
)


# from geom_cairo import ctx
def setup(w: float, h: float, clear_color=[0, 0, 100], range=[0, 1]):
    """
    Setup canvas for drawing
    - w: width (float)
    - h: height (float)
    """
    _setup(w, h, alias=cairo.Antialias.BEST)
    clear_canvas(clear_color)
    set_range(range[0], range[1])


def setup_dpi(size=[8.5, 11], ppi=300, clear_color=[0, 0, 100]):
    """
    Setup for printing
    - size: paper size (i.e. A4 = [8.5" x 11"])
    - dpi: display pixel density (TODO: line widths should adjust accordingly)
    """
    w, h = size
    _setup(w * ppi, h * ppi)
    clear_canvas(clear_color)
    set_bounds(0, 0, w, h)
    x, y, w, h = get_canvas_bounds()
    return (x, y), (w, h)


def clear(clear_color):
    # draw a rect over the bounds
    x, y, w, h = get_canvas_bounds()
    rect(x, y, w, h, attribs={"fill": clear_color})


def draw(dat, attribs=None):
    # if an array of objects, run draw for each
    if isinstance(dat, np.ndarray) or type(dat) == list:
        for item in dat:
            # is it a tuple or list of floats?
            # draw it as a point
            if (
                (type(item) == list or isinstance(item, np.ndarray))
                and len(item) == 2
                and isinstance(item[0], float)
            ):
                draw(Point(item), attribs=attribs)
            else:
                draw(item, attribs=attribs)
        return

    # invoke attribs function if it exists
    attrs = dat.attribs if hasattr(dat, "attribs") else attribs

    if isinstance(dat, Rect):
        rect(dat.x, dat.y, dat.w, dat.h, attribs=attrs)

    elif isinstance(dat, Circle):
        x, y = dat.center
        circle(x, y, dat.r, attribs=attrs)

    elif isinstance(dat, Arc):
        x, y = dat.center
        start, end = dat.theta
        arc(x, y, dat.r, start, end, attribs=attrs)

    elif isinstance(dat, Ellipse):
        x, y = dat.center
        r0, r1 = dat.r
        ellipse(x, y, r0, r1, dat.theta, attribs=attrs)

    elif isinstance(dat, Polygon):
        path(dat.points.tolist(), closed=True, attribs=attrs)

    elif isinstance(dat, Polyline):
        path(dat.points.tolist(), closed=False, attribs=attrs)

    elif isinstance(dat, Line):
        p0, p1 = dat.points
        x0, y0 = p0
        x1, y1 = p1
        line(x0, y0, x1, y1, attribs=attrs)

    elif isinstance(dat, Cubic):
        p0, p1, p2, p3 = dat.points
        curve(p0, p1, p2, p3, attribs=attribs)

    elif isinstance(dat, Point):
        point(dat.pt, attribs=attrs)

    elif hasattr(dat, "points"):
        for pt in dat.points:
            point(pt, attribs=attrs)

    else:
        point(dat, attribs=attrs)


def slap_texture(path):
    import cairo

    surf = cairo.ImageSurface.create_from_png(path)
    paint_surface(surf, cairo.OPERATOR_OVERLAY)


def write_out():
    from datetime import datetime

    surface_out().save(
        "/Users/kylestew/Downloads/"
        + datetime.now().strftime("%Y%m%d-%H%M%S")
        + ".png"
    )


def dump():
    return surface_to_numpy_array()


def display():
    return surface_out()
