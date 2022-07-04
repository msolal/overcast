from overcast.datasets.jasmin0 import JASMINLowResolution
from overcast.datasets.jasmin0 import JASMINHighResolution
from overcast.datasets.jasmin0 import JASMINDailyLowResolution
from overcast.datasets.jasmin0 import JASMINDailyHighResolution

from overcast.datasets.jasmin import JASMIN
from overcast.datasets.jasmin import JASMINDaily
from overcast.datasets.jasmin import JASMINCombo

from overcast.datasets.synthetic import CATE
from overcast.datasets.synthetic import DoseResponse

DATASETS = {
    "jasmin-lr": JASMINLowResolution,
    "jasmin-hr": JASMINHighResolution,
    "jasmin-daily-lr": JASMINDailyLowResolution,
    "jasmin-daily-hr": JASMINDailyHighResolution,
    "jasmin": JASMIN, 
    "jasmin-daily": JASMINDaily,
    "jasmin-combo": JASMINCombo,
    "cate": CATE,
    "dose-response": DoseResponse,
}
