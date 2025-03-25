from strategies import bayesiansearch, gridsearch, randomsearch

def get_hpo_strategy(s : str):
    if s == "bayesian":
        return BayesianSearch()
    elif s == "random":
        return RandomSearch()
    elif s == "grid":
        return GridSearch()
