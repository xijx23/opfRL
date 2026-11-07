"""verl entry point = `verl.trainer.main_ppo` + SIRL Partial-KL.

Takes the same hydra overrides as `python -m verl.trainer.main_ppo` (see configs/verl_*.sh).
Environment:
    OPF_PARTIAL_KL=1|0      partial KL (SIRL) or the stock full-KL penalty
    OPF_PARTIAL_KL_TAG      tag where the KL region starts (default: matlab)
    OPF_VERIFIER_URL        MATLAB verifier used by the reward
    OPF_REWARD_STAGE=1|2    SIRL reward stage
    OPF_RAY_OBJECT_STORE_GB Ray object store size (small machines)
"""
import os
from pathlib import Path

import hydra
import ray
import verl
from verl.trainer import main_ppo
from verl.trainer.constants_ppo import get_ppo_ray_runtime_env

VERL_CONFIG_DIR = str(Path(verl.__file__).parent / "trainer" / "config")
_FORWARDED_ENV = ("OPF_PARTIAL_KL", "OPF_PARTIAL_KL_TAG", "OPF_VERIFIER_URL", "OPF_REWARD_STAGE")


@hydra.main(config_path=VERL_CONFIG_DIR, config_name="ppo_trainer", version_base=None)
def main(config):
    runtime_env = get_ppo_ray_runtime_env()
    env_vars = runtime_env.setdefault("env_vars", {})
    env_vars["OPF_TOKENIZER_PATH"] = config.actor_rollout_ref.model.path
    env_vars.update({k: v for k, v in os.environ.items() if k in _FORWARDED_ENV or k.startswith("NCCL_")})
    runtime_env["worker_process_setup_hook"] = "trainers.verl_sirl.partial_kl_patch.install"
    store_gb = os.environ.get("OPF_RAY_OBJECT_STORE_GB")  # Ray reserves 30% of RAM by default
    ray.init(runtime_env=runtime_env, num_cpus=config.ray_init.num_cpus,
             object_store_memory=int(float(store_gb) * 1024**3) if store_gb else None)
    main_ppo.run_ppo(config)


if __name__ == "__main__":
    main()
