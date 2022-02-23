using JLBoost
using Test
using DataFrames
using Random

X = rand(100, 50)
    
#y = 1.0*(rand(100) .> 0.5)
y = (sum(X', dims=[1]).>25)[1,:];

df = DataFrame(X, :auto);
df[!,"y"] = y;

xgb = jlboost(df, "y"; verbose=false);