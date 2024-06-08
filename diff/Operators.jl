

# flatten(x::GraphNode) = BroadcastedOperator(flatten, x)
# forward(::BroadcastedOperator{typeof(flatten)}, x) = reshape(x, length(x))
# backward(::BroadcastedOperator{typeof(flatten)}, x, g) = tuple(reshape(g, size(x)))

# recurrence(x::GraphNode, h::GraphNode, w_x::GraphNode, w_h::GraphNode) = let

    function tanh_fun(x)
        return (exp.(x) - exp.(-x))/(exp.(x) + exp.(-x))
    end

recurrence(w_x::GraphNode, w_h::GraphNode, x::GraphNode, h::GraphNode) =
    let
        return BroadcastedOperator(recurrence, w_x, w_h, x, h)
    end
forward(::BroadcastedOperator{typeof(recurrence)}, w_x, w_h, x, h) =
    let
        @show "FORWARD RECURRENCE"
        res = w_x * x .+ w_h * h
        return res
    end
backward(::BroadcastedOperator{typeof(recurrence)}, w_x, w_h, x, h, g) =
    let
        @show "BACKWARD RECURRENCE"
        g_wx = g * x'
        g_wh = g * h'
        return tuple(g_wx, g_wh)
    end



dense(x::GraphNode, w::GraphNode) = BroadcastedOperator(dense, x, w)
forward(::BroadcastedOperator{typeof(dense)}, x, w) =
    let
        @show "FORWARD DENSE"
        return w * x
    end
backward(::BroadcastedOperator{typeof(dense)}, x, w, g) =
    let
        @show "BACKWARD DENSE"
        tuple(w' * g, g * x', g)
    end


tanh(x::GraphNode) = BroadcastedOperator(tanh, x)
forward(::BroadcastedOperator{typeof(tanh)}, x) = let
    println("TANH X")
    @show x
    println("CONVERTED TO TANH X")
    @show tanh_fun.(x)
    return tanh_fun.(x)
end
backward(::BroadcastedOperator{typeof(tanh)}, x, g) = let
    return tuple(g .* (-(tanh_fun.(x) .^ 2) .+ 1)) # sprobowac z liczbami dualnymi pozniej
end
# relu(x::GraphNode) = BroadcastedOperator(relu, x)
# forward(::BroadcastedOperator{typeof(relu)}, x) = return max.(x, zero(x))
# backward(::BroadcastedOperator{typeof(relu)}, x, g) = return tuple(g .* (x .> 0))

identity(x::GraphNode) = BroadcastedOperator(identity, x)
forward(::BroadcastedOperator{typeof(identity)}, x) = x
backward(::BroadcastedOperator{typeof(identity)}, x, g) = tuple(g)

cross_entropy_loss(y_hat::GraphNode, y::GraphNode) = BroadcastedOperator(cross_entropy_loss, y_hat, y)
forward(::BroadcastedOperator{typeof(cross_entropy_loss)}, y_hat, y) =
    let
        @show "FORWARD CROSS_ENTROPY"
        num_of_clasiffications = 0 # tu bylo global bez wartosci
        num_of_correct_clasiffications = 0 # tu bylo global bez wartosci
        num_of_clasiffications += 1
        if argmax(y_hat) == argmax(y)
            num_of_correct_clasiffications += 1
        end
        y_hat = y_hat .- maximum(y_hat)
        y_hat = exp.(y_hat) ./ sum(exp.(y_hat))
        loss = sum(log.(y_hat) .* y) * -1.0
        return loss
    end
backward(node::BroadcastedOperator{typeof(cross_entropy_loss)}, y_hat, y, g) =
    let
        y_hat = y_hat .- maximum(y_hat)
        y_hat = exp.(y_hat) ./ sum(exp.(y_hat))
        return tuple(g .* (y_hat .- y))
    end

mse(y_hat::GraphNode, y::GraphNode) = BroadcastedOperator(mse, y_hat, y)
forward(::BroadcastedOperator{typeof(mse)}, y_hat, y) =
    let
        squared_error = 0.0
        for (y_def, y_pred) in zip(y_hat, y)
            squared_error += (y_def - y_pred)^2
        end
        @show squared_error
        return squared_error
    end
backward(node::BroadcastedOperator{typeof(mse)}, y_hat, y, g) =
    let
        println("PRED: " * string(y_hat))
        res_mse = g .* 2 * (y_hat .- y)
        return tuple(res_mse)
    end

