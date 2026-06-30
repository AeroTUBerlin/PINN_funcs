from . import PDEs, PINNs, callbacks, data_handling, layers, metrics, models, plotting, training, utils

_MODULES_IN_EXPORT_ORDER = [
	callbacks,
	data_handling,
	layers,
	metrics,
	models,
	PDEs,
	PINNs,
	plotting,
	training,
	utils,
]

_MODULE_EXPORT_NAMES = [
	"callbacks",
	"data_handling",
	"layers",
	"metrics",
	"models",
	"PDEs",
	"PINNs",
	"plotting",
	"training",
	"utils",
]

for _module in _MODULES_IN_EXPORT_ORDER:
	module_all = getattr(_module, "__all__", None)
	if module_all is None:
		module_all = [name for name in vars(_module).keys() if not name.startswith("_")]

	for _name in module_all:
		if _name not in globals():
			globals()[_name] = getattr(_module, _name)

__all__ = _MODULE_EXPORT_NAMES + [
	name for name in globals().keys()
	if not name.startswith("_") and name not in _MODULE_EXPORT_NAMES
]