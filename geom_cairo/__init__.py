# %load_ext helpers.ipython_cairo

from .draw import (
    setup,
    setup_dpi,
    clear,
    draw,
    display,
    slap_texture,
    write_out,
    dump,
)

FILL = "fill"
STROKE = "stroke"
LINE_WIDTH = "line_width"
LINE_CAP = "line_cap"
POINT_SIZE = "point_size"
BLENDMODE = "blendmode"
