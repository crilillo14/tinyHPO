Both random and grid search are straightforward to implement. 

Think of a mountain range. You *could* look at random points and ask, "Is this the highest point? Is that the highest point?".
You could also look at each individual stone in the mountain range and ask, "Is this higher than the highest point I've looked at?".
 
But realistically, no human is that inefficient at inspecting a mountain range. 

Bayesian Optimization is a complicated topic that is extremely useful for combining random search's sparcity and grid search's effectiveness at finding global maxima. 

Check out the [Comparison of High-Dimensional Bayesian Optimization Algorithms on BBOB] (https://arxiv.org/html/2303.00890v3#S3) for a useful paper on the subject. 

Essentially, fitting a gaussian process to loss of trained models with known hyperparameters, and then using that to make subsequent "best guesses" about where to look next, maximizing expected improvement.