import argparse

import yaml

from tenyson.cloud.modal import ModalManager, _is_truthy_env


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Launch one Tenyson Modal job in an isolated local process."
    )
    parser.add_argument("--job-type", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--task-spec", required=True)
    parser.add_argument("--gpu", required=True)
    parser.add_argument("--timeout", type=int, required=True)
    parser.add_argument("--serialized", default="false")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    with open(args.config, "r", encoding="utf-8") as handle:
        config_payload = yaml.safe_load(handle) or {}

    manager = ModalManager(
        gpu=args.gpu,
        timeout=int(args.timeout),
        serialized=_is_truthy_env(args.serialized),
    )
    manager._run_modal_job(
        job_type=args.job_type,
        config_payload=config_payload,
        task_spec=args.task_spec,
    )


if __name__ == "__main__":
    main()
