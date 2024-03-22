import pretty_errors

pretty_errors.configure(
    separator_character="*",
    filename_display=pretty_errors.FILENAME_EXTENDED,
    line_number_first=True,
    display_link=True,
    display_locals=1,
    lines_before=5,
    lines_after=2,
    line_color=pretty_errors.RED + "> ",
)


from xspike.gpu_queuer import *
from xspike.redis_client import *
from xspike.utils import *
from xspike.redis_maintainer import *
from xspike.io import *
from xspike.cmd_utils import *
from xspike.comet_client import *
