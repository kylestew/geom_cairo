import cairo
from numpy import pi

# from helpers.img_samp import Sampler

# print("***USING PROTOTYPE CONTEXT***")


def _setup(w, h, alias=cairo.Antialias.NONE):
    global _w, _h, _sur, _ctx, _one
    _w = int(w)
    _h = int(h)
    _one = 1.0
    SCALE = 2
    _sur = cairo.ImageSurface(cairo.FORMAT_ARGB32, _w * SCALE, _h * SCALE)
    _sur.set_device_scale(SCALE, SCALE)
    _ctx = cairo.Context(_sur)
    _ctx.set_antialias(alias)
    _ctx.set_line_width(1.0)


def set_range(r0, r1):
    global _min_x, _max_x, _min_y, _max_y, _one, _xoff, _yoff, _xscale, _yscale, _r0, _r1

    small_side = 0
    _xoff = 0
    _yoff = 0
    if _w > _h:
        small_side = _h
        _xoff = (_w - _h) / 2
    else:
        small_side = _w
        _yoff = (_h - _w) / 2

    _xscale = small_side / (r1 - r0)
    _yscale = small_side / (r1 - r0)

    _ctx.translate(_xoff, _yoff)
    _ctx.scale(_xscale, -_yscale)
    _ctx.translate(-r0, -r1)
    _r0 = r0
    _r1 = r1

    _min_x = ((0 - _xoff) / _xscale) + r0
    _max_x = ((_w - _xoff) / _xscale) + r0
    _min_y = ((0 - _yoff) / _yscale) + r0
    _max_y = ((_h - _yoff) / _yscale) + r0

    _one = 1.0 / _xscale
    set_line_width(1.0)


def set_bounds(x0, y0, x1, y1):
    global _min_x, _max_x, _min_y, _max_y, _one, _xscale, _yscale

    _rw = x1 - x0
    _rh = y1 - y0

    # print("range", _rw, _rh)

    _xscale = _w / _rw
    _yscale = _h / _rh

    # print("scale", _xscale, _yscale)

    # _ctx.translate(-_w / 2, -_h / 2)
    _ctx.scale(_xscale, _yscale)
    # _ctx.translate(_rw, _rh)

    _min_x = x0
    _max_x = x1
    _min_y = y0
    _max_y = y1

    _one = 1.0 / _xscale
    set_line_width(1.0)


def clear_canvas(c):
    r, g, b, a = _hsv_to_rgb(c)
    _ctx.set_source_rgba(b, g, r, a)
    _ctx.rectangle(0, 0, _w, _h)
    _ctx.fill()


def get_canvas_size():
    return (_w, _h)


def get_canvas_bounds():
    return (_min_x, _min_y, _max_x, _max_y)


def paint_surface(surf, operator):
    global _xscale, _yscale

    _ctx.save()

    _ctx.translate(_r0, _r1)
    _ctx.scale(1.0 / _xscale, -1.0 / _yscale)
    _ctx.translate(-_xoff, -_yoff)

    _ctx.set_operator(operator)
    _ctx.set_source_surface(surf)
    _ctx.paint()

    _ctx.restore()


def surface_out():
    """
    Creates a scaled surface (original size) and draws the scaled up surface into it
    Used to provide anti-aliased output

    https://pillow.readthedocs.io/en/stable/reference/Image.html
    """
    from PIL import Image
    import PIL

    # TODO: Red and Blue are swapped in output
    # img = Image.frombytes(
    img = Image.frombuffer(
        "RGBA",
        (_sur.get_width(), _sur.get_height()),
        _sur.get_data(),
        # _sur.get_data().tobytes(),
        "raw",
        "RGBA",
        0,
        1,
    )
    # TODO: resampling modes matter depending on what you're doing!?
    return img.resize((_w, _h), resample=PIL.Image.Resampling.LANCZOS)


def surface_to_numpy_array():
    import numpy as np

    # Convert the buffer to a NumPy array
    width = _sur.get_width()
    height = _sur.get_height()
    np_array = np.frombuffer(_sur.get_data(), dtype=np.uint8)
    return np_array.reshape((height, width, 4))


def _prep_sampler(filename):
    import matplotlib.image as mpimg
    from lib.field import Field

    global _sampler

    img = mpimg.imread(filename)
    height, width, depth = img.shape

    field = Field(img)
    _sampler = field.fn([[_min_x, _max_x], [_min_y, _max_y]])

    return width, height


def sample_point(x, y):
    # map coords to img

    return _sampler[y, x]


def apply_attribs(attribs):
    if attribs == None:
        set_color([0, 0, 0, 1])
        _ctx.stroke()
        return

    if "line_width" in attribs:
        set_line_width(attribs["line_width"])
    if "line_cap" in attribs:
        set_line_cap(attribs["line_cap"])
    else:
        set_line_cap(cairo.LINE_CAP_ROUND)

    if "fill" in attribs:
        set_color(attribs["fill"])
        _ctx.fill_preserve()
    if "stroke" in attribs:
        set_color(attribs["stroke"])
        _ctx.stroke()

    if "blendmode" in attribs:
        set_operator(attribs["blendmode"])

    # clear the path
    _ctx.new_path()


def _hsv_to_rgb(c):
    # Need to convert HSBa to RBAa
    import colorsys

    h = c[0] / 360.0
    s = c[1] / 100.0
    v = c[2] / 100.0
    a = 1.0
    if len(c) == 4:
        a = c[3]
    r, g, b = colorsys.hsv_to_rgb(h, s, v)
    return r, g, b, a


def set_color(c):
    r, g, b, a = _hsv_to_rgb(c)
    _ctx.set_source_rgba(b, g, r, a)


def set_operator(op):
    _ctx.set_operator(op)


def set_line_width(width):
    _ctx.set_line_width(_one * width)


def set_line_cap(opt):
    # TODO: may need to convert int to symbol
    _ctx.set_line_cap(opt)
    _ctx.set_line_join(opt)


def line(x0, y0, x1, y1, attribs=None):
    _ctx.move_to(x0, y0)
    _ctx.line_to(x1, y1)
    apply_attribs(attribs)


def rectangle(x, y, w, h, attribs=None):
    _ctx.rectangle(x, y, w, h)
    apply_attribs(attribs)


def point(xy, attribs=None):
    x, y = xy[0], xy[1]  # Extract the x and y coordinates

    pt_size = 1.0
    if attribs != None and "point_size" in attribs.keys():
        pt_size = attribs["point_size"]

    # r = _one * pt_size / 2.0
    # _ctx.rectangle(x - r, y - r, r * 2, r * 2)

    r = _one * pt_size
    _ctx.arc(x, y, r, 0, 2 * pi)

    apply_attribs(attribs)


def circle(x, y, r, attribs=None):
    _ctx.arc(x, y, r, 0, 2 * pi)
    apply_attribs(attribs)


def ellipse(x, y, r0, r1, theta, attribs=None):
    _ctx.save()

    _ctx.translate(x, y)
    _ctx.rotate(theta)
    _ctx.scale(r0, r1)
    _ctx.translate(-x, -y)

    _ctx.new_path()
    _ctx.arc(x, y, 1.0, 0, 2 * pi)
    _ctx.restore()

    apply_attribs(attribs)


def path(xy, closed=False, attribs=None):
    from numpy import array

    xys = array(xy)
    _ctx.move_to(*xys[0, :])
    for p in xys:
        _ctx.line_to(*p)

    if closed:
        _ctx.close_path()

    apply_attribs(attribs)


def arc(cx, cy, r, start, end, attribs=None):
    _ctx.arc(cx, cy, r, start, end)
    apply_attribs(attribs)


def curve(xy, attribs=None):
    """
    https://pycairo.readthedocs.io/en/latest/reference/context.html#cairo.Context.curve_to
    """
    # Check if the input has enough points
    if len(xy) < 4:
        raise ValueError(
            "At least 4 control points are required to draw a cubic Bézier curve."
        )

    # Start drawing from the first point
    _ctx.move_to(xy[0][0], xy[0][1])

    # Iterate over the array in chunks of 4 points
    for i in range(
        0, len(xy) - 3, 3
    ):  # Step by 3 to overlap the end point with the next segment's start point
        _ctx.curve_to(
            xy[i + 1][0],
            xy[i + 1][1],
            xy[i + 2][0],
            xy[i + 2][1],
            xy[i + 3][0],
            xy[i + 3][1],
        )

    # Wrap around: connect the last segment back to the first point
    _ctx.curve_to(
        xy[-2][0],
        xy[-2][1],  # Second-to-last control point
        xy[-1][0],
        xy[-1][1],  # Last control point
        xy[0][0],
        xy[0][1],  # End at the first point to close the curve
    )

    # Apply any additional attributes if provided
    apply_attribs(attribs)
