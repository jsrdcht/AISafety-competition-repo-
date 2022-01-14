args_resnet = {
    'epochs': 200,
    'optimizer_name': 'SGD',
    'optimizer_hyperparameters': {
        'lr': 0.03,
        'momentum': 0.9,
        'weight_decay': 1e-3
    },
    'scheduler_name': 'CosineAnnealingLR',
    'scheduler_hyperparameters': {
        'T_max': 200
    },
    'batch_size': 64,
}
args_densenet = {
    'epochs': 200,
    'optimizer_name': 'SGD',
    'optimizer_hyperparameters': {
        'lr': 0.03,
        'momentum': 0.9,
        'weight_decay': 1e-3
    },
    'scheduler_name': 'CosineAnnealingLR',
    'scheduler_hyperparameters': {
            'T_max': 200
        },
    'batch_size': 64,
}
