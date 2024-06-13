function tanh_fun(x)
    return (exp.(x) - exp.(-x)) / (exp.(x) + exp.(-x))
end

initial_recurrence(w_i::GraphNode, w_h::GraphNode, i::GraphNode, h::GraphNode, b::GraphNode) =
    let
        return BroadcastedOperator(recurrence, w_i, w_h, b, i, h)
    end
recurrence(w_i::GraphNode, w_h::GraphNode, i::GraphNode, operator::BroadcastedOperator, bias::GraphNode) =
    let
        return BroadcastedOperator(recurrence, w_i, w_h, bias, operator, i)
    end
forward(::BroadcastedOperator{typeof(recurrence)}, w_x, w_h, b, x, h) =
    let
        if size(w_x)[2] != size(x)[1]
            buffer = x
            x = h
            h = buffer
        end
        return w_x * x .+ w_h * h .+ b
    end
backward(::BroadcastedOperator{typeof(recurrence)}, w_x, w_h, b, x, h, g) =
    let
        if size(h)[1] == 196
            buffer = x
            x = h
            h = buffer
        end
        return tuple(g * x', g * h', g, w_h * g)
    end


dense(x::GraphNode, w::GraphNode) = BroadcastedOperator(dense, x, w)
forward(::BroadcastedOperator{typeof(dense)}, x, w) =
    let
        return w * x
    end
backward(::BroadcastedOperator{typeof(dense)}, x, w_out, g) =
    let
        tuple(w_out' * g, g * x')
    end


tanh(x::GraphNode) = BroadcastedOperator(tanh, x)
forward(::BroadcastedOperator{typeof(tanh)}, x) =
    let
        return tanh_fun.(x)
    end
backward(::BroadcastedOperator{typeof(tanh)}, x, g) =
    let
        return tuple(g .* (-(tanh_fun.(x) .^ 2) .+ 1))
    end

identity_transpose(x::GraphNode) = BroadcastedOperator(identity_transpose, x)
forward(::BroadcastedOperator{typeof(identity_transpose)}, x) = let 
    x'    
end


backward(::BroadcastedOperator{typeof(identity_transpose)}, x, g) = let
    return tuple(g)
end

cross_entropy_loss(y_hat::GraphNode, y::GraphNode) = BroadcastedOperator(cross_entropy_loss, y_hat, y)
forward(::BroadcastedOperator{typeof(cross_entropy_loss)}, y_hat, y) =
    let
		global num_of_clasiffications
        global num_of_correct_clasiffications
        num_of_clasiffications += 1
        y_hat = y_hat'
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
        y_hat = y_hat'
        y_hat = y_hat .- maximum(y_hat)
        y_hat = exp.(y_hat) ./ sum(exp.(y_hat))
        return tuple(g .* (y_hat .- y))
    end