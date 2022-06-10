from overcast.datasets.jasmin import JASMINLowResolution
from overcast.datasets.jasmin import JASMINHighResolution
from overcast.datasets.jasmin import JASMINDailyLowResolution
from overcast.datasets.jasmin import JASMINDailyHighResolution

from overcast.datasets.synthetic import CATE
from overcast.datasets.synthetic import DoseResponse

DATASETS = {
    "jasmin-lr": JASMINLowResolution,
    "jasmin-hr": JASMINHighResolution,
    "jasmin-daily-lr": JASMINDailyLowResolution,
    "jasmin-daily-hr": JASMINDailyHighResolution,
    "cate": CATE,
    "dose-response": DoseResponse,
}
