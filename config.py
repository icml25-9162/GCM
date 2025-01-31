config = {
    "adult": {
        "use_adam": 1,
        "batch_size": 256,
        "lr": 0.001,
        "decay_r": 0.99,
        "max_epoch": 40,
        "hidden_dim": 16,
        "z_dim": 4,
        "sample_num": 32,
        "beta": 0,
        "loss_type": 0
    },
    "compas": {
        "use_adam": 1,
        "batch_size": 64,
        "lr": 0.01,
        "decay_r": 0.99,
        "max_epoch": 200,
        "hidden_dim": 6,
        "z_dim": 6,
        "sample_num": 32,
        "beta": 0.5,
        "loss_type": 0
    },
    "diabetes": {
        "use_adam": 0,
        "batch_size": 256,
        "lr": 0.3,
        "decay_r": 0.97,
        "max_epoch": 20,
        "hidden_dim": 16,
        "z_dim": 4,
        "sample_num": 32,
        "beta": 0,
        "loss_type": 0
    },
    "blog": {
        "use_adam": 1,
        "batch_size": 256,
        "lr": 0.001,
        "decay_r": 0.99,
        "max_epoch": 40,
        "hidden_dim": 16,
        "z_dim": 4,
        "sample_num": 32,
        "beta": 1,
        "loss_type": 1
    },
    "loan": {
        "use_adam": 1,
        "batch_size": 512,
        "lr": 0.003,
        "decay_r": 0.99,
        "max_epoch": 20,
        "hidden_dim": 16,
        "z_dim": 4,
        "sample_num": 32,
        "beta": 1,
        "loss_type": 0
    },
    "auto": {
        "use_adam": 1,
        "batch_size": 16,
        "lr": 0.01,
        "decay_r": 0.99,
        "max_epoch": 120,
        "hidden_dim": 6,
        "z_dim": 4,
        "sample_num": 32,
        "beta": 1,
        "loss_type": 2
    },

}