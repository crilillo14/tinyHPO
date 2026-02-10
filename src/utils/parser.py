

"""unload a user config dict"""


# allowed keys

keys = {"lr", "m", "dropout_p"}

acceptable_pseudonyms = {"lr" : {"learning rate", "learningrate", "learning_rate"},
                         "m" : {"momentum"},
                         "dropout_p" : {"p", "dropout_probability", "dropout_prob"}}


