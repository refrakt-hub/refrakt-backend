"""Validation and normalization helpers for generated YAML configs."""

from __future__ import annotations

from dataclasses import dataclass, field
import functools
import inspect
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, MutableMapping, Optional, Tuple

import yaml

from config import get_settings


@dataclass(frozen=True)
class ParamSpec:
    """Description of a constructor parameter."""

    name: str
    kind: inspect._ParameterKind
    has_default: bool
    allows_any: bool = False


@dataclass(frozen=True)
class ModelSpec:
    """Metadata describing a registered model."""

    name: str
    parameters: Dict[str, ParamSpec]
    allows_var_kwargs: bool
    allows_var_args: bool

    @property
    def allowed_parameter_names(self) -> Iterable[str]:
        return self.parameters.keys()


@dataclass
class ValidationWarning:
    message: str


@dataclass
class ValidationError(Exception):
    """Error raised when configuration cannot be validated."""

    message: str
    details: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        super().__init__(self.message)

    def __str__(self) -> str:  # pragma: no cover - trivial
        if not self.details:
            return self.message
        details = "\n".join(self.details)
        return f"{self.message}\n{details}"


class UnsupportedModelError(ValidationError):
    """Raised when the requested model is not part of the supported allow-list."""


class FutureModelError(ValidationError):
    """Raised when the requested model is not yet available in refrakt_core."""


@dataclass
class ValidationResult:
    config: Dict[str, Any]
    warnings: list[ValidationWarning] = field(default_factory=list)
    used_fallback: bool = False


@functools.lru_cache(maxsize=1)
def _load_model_specs() -> Dict[str, ModelSpec]:
    """Introspect the refrakt_core model registry for valid parameters."""

    from refrakt_core.registry.model_registry import MODEL_REGISTRY, _import_models

    _import_models()

    specs: Dict[str, ModelSpec] = {}
    for model_name, model_cls in MODEL_REGISTRY.items():
        signature = inspect.signature(model_cls.__init__)
        parameters: Dict[str, ParamSpec] = {}
        allows_var_kwargs = False
        allows_var_args = False

        for param in signature.parameters.values():
            if param.name == "self":
                continue
            if param.kind is inspect.Parameter.VAR_KEYWORD:
                allows_var_kwargs = True
                continue
            if param.kind is inspect.Parameter.VAR_POSITIONAL:
                allows_var_args = True
                continue
            parameters[param.name] = ParamSpec(
                name=param.name,
                kind=param.kind,
                has_default=param.default is not inspect._empty,
            )

        specs[model_name] = ModelSpec(
            name=model_name,
            parameters=parameters,
            allows_var_kwargs=allows_var_kwargs,
            allows_var_args=allows_var_args,
        )

    return specs


class ConfigValidator:
    """Validate generated configuration dictionaries."""

    _SUPPORTED_MODEL_TEMPLATES: Dict[str, str] = {
        "autoencoder": "configs/autoencoder.yaml",
        "resnet18": "configs/resnet.yaml",
        "resnet50": "configs/resnet.yaml",
        "resnet101": "configs/resnet.yaml",
        "resnet152": "configs/resnet.yaml",
        "convnext": "configs/convnext.yaml",
    }

    _SUPPORTED_MODELS = frozenset(_SUPPORTED_MODEL_TEMPLATES.keys())

    _EXTRA_ALIAS_MAPPING: Dict[str, str] = {
        # Vision transformer variants
        "vision transformer": "vit",
        "vision-transformer": "vit",
        "vision transformers": "vit",
        "vision-transformers": "vit",
        "vit": "vit",
        # GAN / SRGAN variants
        "gan": "srgan",
        "gans": "srgan",
        "sr gan": "srgan",
        "sr-gan": "srgan",
        "srgan": "srgan",
        "generative adversarial network": "srgan",
        "generative adversarial networks": "srgan",
        # Autoencoder variants (supported)
        "variational autoencoder": "autoencoder",
        "variational auto-encoder": "autoencoder",
        "vae": "autoencoder",
        "simple autoencoder": "autoencoder",
        # ResNet common aliases
        "resnet": "resnet18",
        "resnet-18": "resnet18",
        "resnet-50": "resnet50",
        "resnet-101": "resnet101",
        "resnet-152": "resnet152",
        # ConvNeXt variants
        "convnext": "convnext",
        "conv-next": "convnext",
        # Other frequently requested but currently unsupported models
        "efficientnet": "efficientnet",
        "efficient net": "efficientnet",
        "efficient-net": "efficientnet",
        "knn": "knn",
        "k-nn": "knn",
        "k nearest neighbor": "knn",
        "k-nearest neighbor": "knn",
        "k nearest neighbours": "knn",
        "k-nearest neighbours": "knn",
        "logistic regression": "logistic_regression",
        "svm": "svc",
        "support vector machine": "svc",
        "support vector machines": "svc",
        "xgboost": "xgboost",
        "random forest": "random_forest",
        "random forests": "random_forest",
        "gradient boosting": "xgboost",
    }

    def __init__(self) -> None:
        self._settings = get_settings()
        self._model_specs = _load_model_specs()
        self._alias_cache: Optional[list[Tuple[str, str]]] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def validate(self, config: Mapping[str, Any]) -> ValidationResult:
        """Validate and normalize a configuration.

        Returns a sanitized version of the config together with warnings.
        May raise ValidationError if the configuration is irreparably wrong.
        """

        if not isinstance(config, Mapping):
            raise ValidationError("Configuration must be a mapping object")

        normalized = self._deepcopy_dict(config)
        warnings: list[ValidationWarning] = []

        self._ensure_required_sections(normalized)

        warning_messages = self._sanitize_model_section(normalized)
        warnings.extend(ValidationWarning(msg) for msg in warning_messages)

        self._validate_runtime_section(normalized)
        self._validate_dataset_section(normalized)
        self._validate_dataloader_section(normalized)
        self._validate_loss_section(normalized)
        self._validate_optimizer_section(normalized)
        self._validate_scheduler_section(normalized)
        self._validate_trainer_section(normalized)

        return ValidationResult(config=normalized, warnings=warnings)

    def fallback_to_template(
        self,
        model_name: str,
        overrides: Optional[Mapping[str, Any]] = None,
    ) -> ValidationResult:
        """Load a canonical template and overlay optional overrides."""

        if model_name not in self._SUPPORTED_MODELS:
            supported = ", ".join(self.supported_model_names())
            raise UnsupportedModelError(
                f'Refrakt does not currently support "{model_name}".',
                [f"Supported models: {supported}"],
            )

        template_path = self._resolve_template_path(model_name)
        if template_path is None:
            raise ValidationError(
                f"No template available for model '{model_name}'.",
            )

        with template_path.open("r", encoding="utf-8") as handle:
            template_cfg = yaml.safe_load(handle)

        if overrides and isinstance(overrides, Mapping):
            merged = self._merge_dict(template_cfg, overrides)
        else:
            merged = template_cfg

        result = self.validate(merged)
        result.used_fallback = True
        return result

    # ------------------------------------------------------------------
    # Section checks
    # ------------------------------------------------------------------
    def _ensure_required_sections(self, config: MutableMapping[str, Any]) -> None:
        for key in ("runtime", "dataset", "dataloader", "model", "loss", "optimizer", "scheduler", "trainer"):
            config.setdefault(key, {})

    def _sanitize_model_section(self, config: MutableMapping[str, Any]) -> list[str]:
        model_section = self._require_mapping(config, "model")
        if model_section is None:
            raise ValidationError("'model' section must be a mapping")

        model_name = model_section.get("name")
        if not isinstance(model_name, str):
            raise ValidationError("'model.name' must be provided as a string")

        if model_name not in self._SUPPORTED_MODELS:
            supported = ", ".join(self.supported_model_names())
            raise UnsupportedModelError(
                f'Refrakt does not currently support "{model_name}".',
                [f"Supported models: {supported}"],
            )

        if model_name not in self._model_specs:
            raise FutureModelError(
                f'Refrakt will support "{model_name}" soon!',
            )

        spec = self._model_specs[model_name]
        params = model_section.get("params")
        if params is None:
            params = {}
            model_section["params"] = params
        if not isinstance(params, MutableMapping):
            raise ValidationError("'model.params' must be a mapping")

        warnings: list[str] = []
        invalid_keys = [
            key for key in list(params.keys())
            if key not in spec.parameters and not spec.allows_var_kwargs
        ]

        for key in invalid_keys:
            del params[key]
            warnings.append(
                f"Removed unsupported parameter '{key}' from model '{model_name}'."
            )

        missing_required = [
            name for name, param_spec in spec.parameters.items()
            if not param_spec.has_default and name not in params
        ]

        if missing_required:
            raise ValidationError(
                f"Model '{model_name}' is missing required parameters: {', '.join(missing_required)}"
            )

        return warnings

    def _validate_runtime_section(self, config: MutableMapping[str, Any]) -> None:
        runtime = config.get("runtime")
        if not isinstance(runtime, MutableMapping):
            raise ValidationError("'runtime' section must be a mapping")
        runtime.setdefault("mode", "pipeline")
        runtime.setdefault("log_type", [])

    def _validate_dataset_section(self, config: MutableMapping[str, Any]) -> None:
        dataset = config.get("dataset")
        if not isinstance(dataset, MutableMapping):
            raise ValidationError("'dataset' section must be a mapping")
        dataset.setdefault("params", {})
        dataset.setdefault("transform", [])

    def _validate_dataloader_section(self, config: MutableMapping[str, Any]) -> None:
        dataloader = config.get("dataloader")
        if dataloader in (None, {}):
            config["dataloader"] = {"params": {}}
            return
        if not isinstance(dataloader, MutableMapping):
            raise ValidationError("'dataloader' section must be a mapping")
        dataloader.setdefault("params", {})

    def _validate_loss_section(self, config: MutableMapping[str, Any]) -> None:
        loss = config.get("loss")
        if not isinstance(loss, MutableMapping):
            raise ValidationError("'loss' section must be a mapping")
        loss.setdefault("params", {})

    def _validate_optimizer_section(self, config: MutableMapping[str, Any]) -> None:
        optimizer = config.get("optimizer")
        if not isinstance(optimizer, MutableMapping):
            raise ValidationError("'optimizer' section must be a mapping")
        optimizer.setdefault("params", {})

    def _validate_scheduler_section(self, config: MutableMapping[str, Any]) -> None:
        scheduler = config.get("scheduler")
        if scheduler is None:
            config["scheduler"] = {"name": None, "params": None}
            return
        if isinstance(scheduler, MutableMapping):
            scheduler.setdefault("name", None)
            scheduler.setdefault("params", None)
            return
        raise ValidationError("'scheduler' section must be a mapping or null")

    def _validate_trainer_section(self, config: MutableMapping[str, Any]) -> None:
        trainer = config.get("trainer")
        if not isinstance(trainer, MutableMapping):
            raise ValidationError("'trainer' section must be a mapping")
        trainer.setdefault("params", {})

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @classmethod
    def supported_model_names(cls) -> list[str]:
        return sorted(cls._SUPPORTED_MODELS)

    def is_registered_model(self, model_name: str) -> bool:
        return model_name in self._model_specs

    def unsupported_model_names(self) -> list[str]:
        return sorted(
            name for name in self._model_specs.keys() if name not in self._SUPPORTED_MODELS
        )

    def find_unsupported_model_reference(self, text: str) -> Optional[Tuple[str, str]]:
        lowered = text.lower()
        for alias, canonical in self._model_alias_pairs():
            if alias in lowered and canonical not in self._SUPPORTED_MODELS:
                return alias, canonical
        return None

    def _model_alias_pairs(self) -> list[Tuple[str, str]]:
        if self._alias_cache is not None:
            return self._alias_cache

        alias_map: Dict[str, str] = {}
        for model_name in self._model_specs.keys():
            normalized = model_name.lower()
            alias_map.setdefault(normalized, model_name)
            alias_map.setdefault(normalized.replace("_", " "), model_name)

        for alias, canonical in self._EXTRA_ALIAS_MAPPING.items():
            alias_map[alias] = canonical

        self._alias_cache = sorted(
            ((alias, canonical) for alias, canonical in alias_map.items() if alias),
            key=lambda pair: len(pair[0]),
            reverse=True,
        )

        return self._alias_cache

    def _resolve_template_path(self, model_name: str) -> Optional[Path]:
        template_rel = self._SUPPORTED_MODEL_TEMPLATES.get(model_name)
        if template_rel is None:
            return None
        candidate = (self._settings.PROJECT_ROOT / template_rel).resolve()
        return candidate if candidate.exists() else None

    @staticmethod
    def _deepcopy_dict(mapping: Mapping[str, Any]) -> Dict[str, Any]:
        result: Dict[str, Any] = {}
        for key, value in mapping.items():
            if isinstance(value, Mapping):
                result[key] = ConfigValidator._deepcopy_dict(value)
            elif isinstance(value, list):
                result[key] = [ConfigValidator._deepcopy_dict(v) if isinstance(v, Mapping) else v for v in value]
            else:
                result[key] = value
        return result

    @staticmethod
    def _merge_dict(base: Any, overrides: Any) -> Any:
        if not isinstance(base, Mapping) or not isinstance(overrides, Mapping):
            return overrides if overrides is not None else base

        merged: Dict[str, Any] = dict(base)
        for key, value in overrides.items():
            if key in merged:
                merged[key] = ConfigValidator._merge_dict(merged[key], value)
            else:
                merged[key] = value
        return merged

    @staticmethod
    def _require_mapping(config: MutableMapping[str, Any], key: str) -> Optional[MutableMapping[str, Any]]:
        value = config.get(key)
        if value is None:
            config[key] = {}
            value = config[key]
        if isinstance(value, MutableMapping):
            return value
        return None


