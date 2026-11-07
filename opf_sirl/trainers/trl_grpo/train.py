"""Entry point of the TRL back-end.

    python -m trainers.trl_grpo.train --config configs/trl_local_smoke.yaml
"""
from dataclasses import dataclass, field

from datasets import load_dataset
from trl import GRPOConfig, ModelConfig, TrlParser, get_peft_config

from trainers.trl_grpo.trainer import PartialKLGRPOTrainer, opf_reward


@dataclass
class DataArgs:
    train_file: str = field(default="data/train.parquet")
    val_file: str = field(default=None)
    kl_tag: str = field(default="matlab", metadata={"help": "partial-KL starts at <kl_tag>"})


def main(argv=None):
    data_args, training_args, model_args = TrlParser((DataArgs, GRPOConfig, ModelConfig)).parse_args_and_config(argv)
    files = {"train": data_args.train_file}
    if data_args.val_file:
        files["validation"] = data_args.val_file
    dataset = load_dataset("parquet", data_files=files)

    trainer = PartialKLGRPOTrainer(
        model=model_args.model_name_or_path,
        reward_funcs=opf_reward,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset.get("validation"),
        peft_config=get_peft_config(model_args),
        kl_tag=data_args.kl_tag,
    )
    trainer.train()
    trainer.save_model(training_args.output_dir)


if __name__ == "__main__":
    main()
