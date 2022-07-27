from overcast.datasets.jasmin import JASMIN
from overcast.datasets.jasmin import JASMINDaily
from overcast.datasets.jasmin import JASMINCombo

from overcast.datasets.synthetic import CATE
from overcast.datasets.synthetic import DoseResponse

DATASETS = {
    "jasmin": JASMIN, 
    "jasmin-daily": JASMINDaily,
    "jasmin-combo": JASMINCombo,
    "cate": CATE,
    "dose-response": DoseResponse,
}
