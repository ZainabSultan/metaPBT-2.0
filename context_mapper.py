# DEFAULTS
import numpy as np


def get_defaults(env_name):

    if env_name == "CARLBipedalWalker":
        feature_defaults = {
            "FPS": 50,
            "SCALE": 30.0,
            "GRAVITY_X": 0,
            "GRAVITY_Y": -10,
            "FRICTION": 2.5,
            "TERRAIN_STEP": 14 / 30.0,
            "TERRAIN_LENGTH": 200,
            "TERRAIN_HEIGHT": 600 / 30 / 4,
            "TERRAIN_GRASS": 10,
            "TERRAIN_STARTPAD": 20,
            "MOTORS_TORQUE": 80,
            "SPEED_HIP": 4,
            "SPEED_KNEE": 6,
            "LIDAR_RANGE": 160 / 30.0,
            "LEG_DOWN": -8 / 30.0,
            "LEG_W": 8 / 30.0,
            "LEG_H": 34 / 30.0,
            "INITIAL_RANDOM": 5,
            "VIEWPORT_W": 600,
            "VIEWPORT_H": 400,
        }
    if env_name == "CARLLunarLander":
            feature_defaults = {
                "FPS": 50,
                "SCALE": 30.0,
                "MAIN_ENGINE_POWER": 13.0,
                "SIDE_ENGINE_POWER": 0.6,
                "INITIAL_RANDOM": 1000.0,
                "GRAVITY_X": 0,
                "GRAVITY_Y": -10,
                "LEG_AWAY": 20,
                "LEG_DOWN": 18,
                "LEG_W": 2,
                "LEG_H": 8,
                "LEG_SPRING_TORQUE": 40,
                "SIDE_ENGINE_HEIGHT": 14.0,
                "SIDE_ENGINE_AWAY": 12.0,
                "VIEWPORT_W": 600,
                "VIEWPORT_H": 400,
            }
    if env_name == 'CARLMountainCar':
        feature_defaults = {'gravity': 0.0025}
    if env_name == 'CARLCartPole':
        feature_defaults = {'length': 0.5, 'tau': 0.02, 'gravity': 9.8}

    return feature_defaults
     

def get_context_id(env_name, feature_value_pair):

    values = np.array([
    0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 
    1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 
    2.1, 2.2
        ])
    value = next(iter(feature_value_pair.values()))
    feature= next(iter(feature_value_pair.keys()))

    defaults = get_defaults(env_name)

    factor = value/defaults[feature] 

    context = np.argmax(values == factor)

    return context+1




     

def get_context(env_name, context_id, context_feature):
    values = np.array([
    0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 
    1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 
    2.1, 2.2
])

    feature_defaults = get_defaults(env_name)
    multiplier_value = values[context_id]
    if context_feature in feature_defaults:
        feature_defaults[context_feature] *= multiplier_value
    return feature_defaults[context_feature]
