from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

from datasets import Dataset


class TaskPlugin(ABC):
    def get_environment_name(self) -> Optional[str]:
        return None

    def list_named_runs(self, run_type: Optional[str] = None) -> Sequence[str]:
        return []

    def get_named_run_type(self, run_name: str) -> Optional[str]:
        return None

    def get_named_run_config_overrides(self, run_name: str) -> Optional[Mapping[str, Any]]:
        return None

    def get_run_config_overrides(
        self,
        run_type: str,
        *,
        variant: Optional[str] = None,
    ) -> Optional[Mapping[str, Any]]:
        return None

    def list_run_variants(self, run_type: str) -> Sequence[str]:
        return []

    @abstractmethod
    def get_sft_dataset(self, config: Dict[str, Any], tokenizer: Any) -> Dataset:
        raise NotImplementedError

    def get_sft_eval_dataset(self, config: Dict[str, Any], tokenizer: Any) -> Optional[Dataset]:
        return None

    def get_sft_formatting_func(self, config: Dict[str, Any], tokenizer: Any) -> Optional[Callable]:
        return None

    def get_sft_data_collator(self, config: Dict[str, Any], tokenizer: Any) -> Optional[Any]:
        return None

    @abstractmethod
    def get_rl_dataset(self, config: Dict[str, Any]) -> Dataset:
        raise NotImplementedError

    @abstractmethod
    def get_reward_funcs(self, config: Dict[str, Any], tokenizer: Any) -> List[Callable]:
        raise NotImplementedError

    def get_rl_callbacks(self, config: Dict[str, Any], tokenizer: Any, output_dir: str) -> List[Any]:
        return []

    @abstractmethod
    def get_eval_dataset(self, config: Dict[str, Any]) -> Dataset:
        raise NotImplementedError

    @abstractmethod
    def compute_metrics(
        self,
        prompts: List[str],
        completions: List[str],
        dataset_rows: Dataset,
        config: Dict[str, Any],
        tokenizer: Any,
    ) -> Dict[str, Any]:
        raise NotImplementedError


class TemplateTaskPlugin(TaskPlugin):
    def __init__(self, *, environment_name: Optional[str] = None) -> None:
        resolved_name = str(environment_name or "").strip()
        self._environment_name = resolved_name or None

    def get_environment_name(self) -> Optional[str]:
        return self._environment_name

    def get_sft_dataset(self, config: Dict[str, Any], tokenizer: Any) -> Dataset:
        del config, tokenizer
        raise ValueError(
            "This task is template-driven. Pass an SFT dataset template to "
            '`exp.sft(..., dataset=...)`.'
        )

    def get_rl_dataset(self, config: Dict[str, Any]) -> Dataset:
        del config
        raise ValueError(
            "This task is template-driven. Pass an RL dataset template to "
            '`exp.rl(..., dataset=...)`.'
        )

    def get_reward_funcs(self, config: Dict[str, Any], tokenizer: Any) -> List[Callable]:
        del config, tokenizer
        raise ValueError(
            "This task is template-driven. Pass an RL reward template to "
            '`exp.rl(..., reward=...)`.'
        )

    def get_eval_dataset(self, config: Dict[str, Any]) -> Dataset:
        del config
        raise ValueError(
            "This task is template-driven. Pass an eval dataset template to "
            '`exp.eval(..., dataset=...)`.'
        )

    def compute_metrics(
        self,
        prompts: List[str],
        completions: List[str],
        dataset_rows: Dataset,
        config: Dict[str, Any],
        tokenizer: Any,
    ) -> Dict[str, Any]:
        del prompts, completions, dataset_rows, config, tokenizer
        raise ValueError(
            "This task is template-driven. Pass an eval metrics template to "
            '`exp.eval(..., metrics=...)`.'
        )
