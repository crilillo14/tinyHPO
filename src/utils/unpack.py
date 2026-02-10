

"""unload a user config dict"""


# allowed keys

keys = {"lr", "m", "dropout_p"}

acceptable_pseudonyms = {"lr" : {"learning rate", "learningrate", "learning_rate"},
                         "m" : {"momentum"},
                         "dropout_p" : {"p", "dropout_probability", "dropout_prob"}}

def unpack_config(config : dict):
    
    for k , v in config.items():
        if k in keys:
            yield k , v
        elif k in acceptable_pseudonyms:
            yield from unpack_config(v)