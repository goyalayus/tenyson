from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional

from datasets import Dataset


class TaskPlugin(ABC):
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
