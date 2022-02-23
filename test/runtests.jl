using JLBoost
using Test
using DataFrames
using Random

# @testset "smoke test" begin
#     df = DataFrame(x=[1,1,1,0,0], y = [1,1,1,0,0])
#     jlboost(df, :y; nrounds=4)
# end

@testset "smoke test" begin
    X = rand(100, 50)
    
    #y = 1.0*(rand(100) .> 0.5)
    y = (sum(X', dims=[1]).>25)[1,:];
    
    df = DataFrame(X, :auto);
    df[!,"y"] = y;

    xgb = jlboost(df, "y"; verbose=false, max_depth = 5);

    y_hat = predict(xgb, DataFrame(X, :auto))
    println(y_hat)

    y_hat = (y_hat .>=0)
    accuracy = sum(y .== y_hat)/length(y)

    ac = AUC(y_hat, df[!,"y"])
    println("Accuracy $(accuracy)")
    println("AUC: $(ac)")
end
