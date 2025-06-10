import os

BASE_PATH = '/home/agmota/ros2_ws/UPM.MAADM.TFM'

class ModelInfo:
    def __init__(self, base_path, model_name, weights_name):
        self.weights_path = os.path.join(base_path, 'weights', weights_name)
        self.json_path = os.path.join(base_path, 'models', f'{model_name}.json')
        self.model_name = model_name

def get_model_info(name):
    models = {
        'cnn': ModelInfo(
            base_path=BASE_PATH,
            model_name='cnn_64_32_1_dense_64',
            weights_name='best_cnn_64_32_1_dense_64_20250604_175528.h5'
        ),
        'lstm': ModelInfo(
            base_path=BASE_PATH,
            model_name='lstm',
            weights_name='best_lstm_20250604_180405.h5'
        ),
        'neural_network': ModelInfo(
            base_path=BASE_PATH,
            model_name='neural_network',
            weights_name='best_neural_network_20250604_174337.h5'
        ),
        'transformer': ModelInfo(
            base_path=BASE_PATH,
            model_name='transformer',
            weights_name='best_transformer_20250604_181401.h5'
        ),        
        'transformer_dql': ModelInfo(
            base_path=BASE_PATH,
            model_name='transformer',
            weights_name='dql_transformer_20250608_170153.h5'
        ),
        'transformer_dql2': ModelInfo(
            base_path=BASE_PATH,
            model_name='transformer',
            weights_name='dql_transformer_20250610_161241.h5'
        ),
        'transformer_dql3': ModelInfo(
            base_path=BASE_PATH,
            model_name='transformer',
            weights_name='dql_transformer_20250610_164428.h5'
        ),
        'transformer_dql4': ModelInfo(
            base_path=BASE_PATH,
            model_name='transformer',
            weights_name='dql_transformer_20250610_180523.h5'
        ),
        'transformer_with_positional_encoding': ModelInfo(
            base_path=BASE_PATH,
            model_name='transformer_with_positional_encoding',
            weights_name='best_transformer_with_positional_encoding_20250604_184529.h5'
        ),
    }

    if name not in models:
        raise ValueError(f"Model '{name}' not found.")
    
    return models[name]