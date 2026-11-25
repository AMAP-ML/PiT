import ml_collections


def d(**kwargs):
    """Helper of creating a config dict."""
    return ml_collections.ConfigDict(initial_dictionary=kwargs)


def get_config():
    config = ml_collections.ConfigDict()

    _image_size = 256
    _train_steps = 1000000

    config.seed = 4567
    config.path = ""
    config.mixed_precision = "bf16"
    config.data_shape = [3, _image_size, _image_size]

    config.train = d(
        total_training_steps=_train_steps,
        ema_rate=0.9999,
        log_interval=10,
        sample_interval=5000,
        save_interval=50000,
        eval_interval=50000,
        gradclip=0.0,
    )

    config.lr_scheduler = d(
        name="warmup",
        warmup_steps=5000,
        total_training_steps=_train_steps,
        min_scale=-1.0,
    )

    config.optimizer = d(
        lr=2e-4,
        weight_decay=0.0,
        betas=(0.9, 0.95),
        fused=True
    )

    config.diffusion = d(
        sigma_data=0.5,
        sigma_max=80.0,
        sigma_min=0.002,
        schedule_sampler="lognormal_continuous",
        sample_param=d(
            p_mean=-1.2,
            p_std=1.6,
        ),
        shift_sigma=True,
    )

    config.model_name = "PiT-XL"
    config.model_params = d(
        image_size=_image_size,
        patch_size=16
    )

    config.dataset = d(
        name="imagenet",
        image_shape=config.data_shape,
        data_dir="",
        batch_size=1024, # global batch size
        num_workers=8,
        cfg_drop_rate=0.1,
        std="imagenet",
        horizontal_flip=True,
        fid_stat_path="",
    )

    config.sample = d(
        sampler="euler",
        sampling_step=50,
        num_samples=100,
        mini_batch_size=100,
        save_sample_traj=False,
        path="",
        cfg_scale=0.0,
        cfg_sigma_low=0.19,
        cfg_sigma_high=1.61,
    )

    return config