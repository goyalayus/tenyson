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
