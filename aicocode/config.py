"""配置层：读取并校验 YAML 配置，得到 providers 列表。"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from .config_validator import (
    ConfigError,
    Protocols,
    VALID_PROTOCOLS,
    DEFAULT_CONTEXT_WINDOW,
    lookup_model_context_window,
    validate_config,
)

_REQUIRED_FIELDS: tuple[str, ...] = ("name", "protocol", "model", "api_key")

_ENV_KEY_MAP = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "openai-compat": "OPENAI_API_KEY",
}

_ENV_VAR_RE = re.compile(r"\$\{([^}]+)\}")

@dataclass
class ProviderConfig:
    """单个供应商配置。"""

    name: str
    protocol: Protocols
    model: str
    api_key: str
    base_url: str | None = None
    default: bool = False
    thinking: bool = False
    context_window: int = 0
    max_output_tokens: int = 0
    # 运行时 cache，存放从 provider 的 /v1/models 端点自动拉取的 context window
    # 通过 set_fetched_context_window() 写入一次；
    _fetched_context_window: int = field(default=0, repr=False)

    def resolve_api_key(self) -> str:
        """
            解析 api_key
        """
        if self.api_key:
            return self.api_key
        env_var = _ENV_KEY_MAP.get(self.protocol, "")
        return os.environ.get(env_var, "")

    def set_fetched_context_window(self, window: int) -> None:
        """
            记录从 provider 自动拉取到的 context window。
        """
        if window > 0:
            self._fetched_context_window = window

    def get_context_window(self) -> int:
        """通过四层 fallback 解析模型的 context window，按优先级从高到低：

          1. 配置文件提供的 context_window（> 0）——显式覆盖，永远优先。
          2. 从 provider 的 /v1/models 端点自动拉取并通过 set_fetched_context_window
             缓存的值（只有 anthropic 协议的 provider 才会设置它；拉取失败或缺失时
             保持为 0 并跳过）。
          3. 内置的「模型名 -> window」映射表（按子串匹配）。
          4. 保守的默认值（claude -> 200000，其他 -> 128000）。
        """
        if self.context_window > 0:
            return self.context_window
        if self._fetched_context_window > 0:
            return self._fetched_context_window
        window = lookup_model_context_window(self.model)
        if window > 0:
            return window
        if "claude" in self.model.lower():
            return DEFAULT_CONTEXT_WINDOW
        return 128_000

    def get_max_output_tokens(self) -> int:
        if self.max_output_tokens > 0:
            return self.max_output_tokens
        if self.thinking:
            return 64000
        return 8192

@dataclass
class AppConfig:
    """
        应用配置：providers 列表 + 初始选定供应商名。
    """
    providers: list[ProviderConfig]

def _load_config_yaml(path: Path) -> AppConfig:
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as e:
        raise ConfigError(f"Failed to parse config {path}: {e}") from e

    validated_config = validate_config(raw)

    providers = [
        ProviderConfig(
            name=p["name"],
            protocol=p["protocol"],
            base_url=p["base_url"],
            model=p["model"],
            api_key=p["api_key"],
            thinking=p["thinking"],
            context_window=p["context_window"],
            max_output_tokens=p["max_output_tokens"],
        )
        for p in validated_config["providers"]
    ]

    return AppConfig(
        providers=providers,
    )

def _merge_existing_config(base: AppConfig, override: AppConfig) -> AppConfig:
    if override.providers:
        base.providers = override.providers

    return base


def load_config(path: Path | None = None) -> AppConfig:
    if path is not None:
        if not path.exists():
            raise ConfigError(f"Config file not found: {path}")
        return _load_config_yaml(path)

    cwd = Path.cwd()
    home = Path.home()
    candidates = [
        home / ".aicocode" / "config.yaml",
        cwd / ".aicocode" / "config.yaml",
        cwd / ".aicocode" / "config.local.yaml",
    ]

    merged_cfg: AppConfig | None = None
    for p in candidates:
        if not p.exists():
            continue
        cfg = _load_config_yaml(p)
        if merged_cfg is None:
            merged_cfg = cfg
        else:
            merged_cfg = _merge_existing_config(merged_cfg, cfg)

    if merged_cfg is None:
        raise ConfigError(
            "No config file found. Expected .mewcode/config.yaml "
            "in project or ~/.mewcode/config.yaml"
        )
    return merged_cfg