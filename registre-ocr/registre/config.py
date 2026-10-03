"""Configuration : config.toml à la racine (facultatif) + variables d'environnement.

Exemple de config.toml :

    [vlm]
    base_url = "http://localhost:11434/v1"   # Ollama ; LM Studio : http://localhost:1234/v1
    model = "qwen3-vl:8b-instruct"
    batch = 8
    scale = 2.0

    [pipeline]
    templates = "templates"
    out = "out"
"""
from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class VLMConfig:
    base_url: str = "http://localhost:11434/v1"
    model: str = "qwen3-vl:8b-instruct"
    batch: int = 8
    scale: float = 2.0
    timeout: float = 240.0
    allow_remote: bool = False
    backend: str = "auto"      # "ollama" (API native, permet think=false) | "openai" | "auto"
    think: bool = False        # raisonnement du modèle : coupé (plus rapide, JSON non tronqué)
    parallel: int = 1          # appels simultanés au serveur : 1 sur Mac, 8-16 avec vLLM sur GPU


@dataclass
class Config:
    vlm: VLMConfig = field(default_factory=VLMConfig)
    templates: str = "templates"
    out: str = "out"


def load(path: str | None = None) -> Config:
    path = path or os.environ.get("REGISTRE_CONFIG", "config.toml")
    cfg = Config()
    p = Path(path)
    if p.exists():
        data = tomllib.loads(p.read_text())
        for k, v in data.get("vlm", {}).items():
            setattr(cfg.vlm, k, v)
        for k, v in data.get("pipeline", {}).items():
            setattr(cfg, k, v)
    cfg.vlm.base_url = os.environ.get("REGISTRE_VLM_URL", cfg.vlm.base_url)
    cfg.vlm.model = os.environ.get("REGISTRE_VLM_MODEL", cfg.vlm.model)
    return cfg
