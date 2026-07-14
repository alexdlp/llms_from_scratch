from omegaconf import DictConfig
from transformer_from_scratch import create_pipeline
from transformer_from_scratch.logger import logger
from transformer_from_scratch.utils.config_utils import (
    compose_config,
    export_args_to_env,
    load_environment,
    parse_args,
)
from transformer_from_scratch.utils.io_utils import clean_up


def run_training_pipeline(cfg: DictConfig):

    pipeline = create_pipeline(cfg)

    try:
        pipeline.fit()
    except Exception as ex:
        logger.error(f"❌ Exception occurred: {ex}", exc_info=True)
    finally:
        clean_up()


def main():
    # 1. Parse external flags and OmegaConf overrides
    args, config_overrides = parse_args()

    # 2. Export CLI args to env 
    export_args_to_env(args)

    # 3. Load .env environment
    load_environment()

    # 4. Compose and launch the training pipeline
    cfg = compose_config(overrides=config_overrides)

    # 5. Run the pipeline 
    run_training_pipeline(cfg=cfg)


if __name__ == "__main__":
    main()
