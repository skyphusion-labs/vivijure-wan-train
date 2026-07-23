"""Bundle + job contract for the Wan train satellite (subset of vivijure-backend contract)."""
from __future__ import annotations

import json
import tarfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

SLOT_IDS = ("A", "B", "C", "D")


def _str(v: Any, default: str = "") -> str:
    return v if isinstance(v, str) else default


@dataclass
class Scene:
    prompt: str
    id: str | None = None
    character_slots: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, d: dict[str, Any], index: int) -> "Scene":
        slots = [s for s in (d.get("character_slots") or []) if s in SLOT_IDS]
        return cls(
            prompt=_str(d.get("prompt")),
            id=_str(d.get("id")) or f"shot_{index + 1:02d}",
            character_slots=slots,
        )


@dataclass
class Storyboard:
    title: str
    scenes: list[Scene]
    use_characters: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Storyboard":
        scenes = [Scene.from_dict(s, i) for i, s in enumerate(d.get("scenes") or []) if isinstance(s, dict)]
        if not scenes:
            raise ValueError("storyboard.json has no scenes")
        use_chars = [s for s in (d.get("use_characters") or []) if s in SLOT_IDS]
        return cls(title=_str(d.get("title"), "untitled"), scenes=scenes, use_characters=use_chars)

    @classmethod
    def from_yaml(cls, text: str) -> "Storyboard":
        return cls.from_dict(yaml.safe_load(text) or {})


@dataclass
class Character:
    slot: str
    name: str
    prompt: str
    ref_paths: list[Path] = field(default_factory=list)


@dataclass
class Cast:
    characters: dict[str, Character]

    @classmethod
    def from_registry(cls, registry: dict[str, Any]) -> "Cast":
        raw = registry.get("characters") or {}
        out: dict[str, Character] = {}
        for slot, c in raw.items():
            if slot not in SLOT_IDS or not isinstance(c, dict):
                continue
            out[slot] = Character(slot=slot, name=_str(c.get("name"), slot), prompt=_str(c.get("prompt")))
        return cls(characters=out)


@dataclass
class Bundle:
    root: Path
    storyboard: Storyboard
    cast: Cast

    @classmethod
    def extract(cls, tar_path: Path, dest: Path) -> "Bundle":
        dest.mkdir(parents=True, exist_ok=True)
        with tarfile.open(tar_path, "r:gz") as tf:
            _safe_extract(tf, dest)
        sb_path = dest / "storyboard.yaml"
        if not sb_path.is_file():
            raise FileNotFoundError(f"bundle is missing storyboard.yaml at {sb_path}")
        storyboard = Storyboard.from_yaml(sb_path.read_text(encoding="utf-8"))
        reg_path = dest / "characters" / "registry.json"
        cast = Cast.from_registry(json.loads(reg_path.read_text(encoding="utf-8")) if reg_path.is_file() else {})
        refs_root = dest / "characters" / "refs"
        for slot, char in cast.characters.items():
            slot_dir = refs_root / slot
            if slot_dir.is_dir():
                char.ref_paths = sorted(
                    p for p in slot_dir.iterdir()
                    if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp")
                )
        return cls(root=dest, storyboard=storyboard, cast=cast)


def _safe_extract(tf: tarfile.TarFile, dest: Path) -> None:
    dest = dest.resolve()
    for member in tf.getmembers():
        if member.issym() or member.islnk():
            raise ValueError(f"unsafe link in bundle: {member.name}")
        if member.isdev() or member.isfifo():
            raise ValueError(f"unsafe special file in bundle: {member.name}")
        target = (dest / member.name).resolve()
        if not target.is_relative_to(dest):
            raise ValueError(f"unsafe path in bundle: {member.name}")
    tf.extractall(dest)


@dataclass
class TrainRequest:
    """Minimal job body the control plane submits to the Wan train endpoint."""
    action: str
    project: str
    bundle_key: str
    pretrained_loras: dict[str, str] = field(default_factory=dict)
    model_family: str = "wan"

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "TrainRequest":
        return cls(
            action=_str(d.get("action"), "train_lora"),
            project=_str(d.get("project"), "untitled"),
            bundle_key=_str(d.get("bundle_key")),
            pretrained_loras=d.get("pretrained_loras") if isinstance(d.get("pretrained_loras"), dict) else {},
            model_family=_str(d["model_family"], "wan") if "model_family" in d else "wan",
        )


@dataclass
class TrainResult:
    project: str
    lora: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"project": self.project, "lora": self.lora, "output_key": None, "seconds": None,
                "has_audio": False, "audio_missing": False, "keyframes": [], "state_key": None}
