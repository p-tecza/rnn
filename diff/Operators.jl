

# flatten(x::GraphNode) = BroadcastedOperator(flatten, x)
# forward(::BroadcastedOperator{typeof(flatten)}, x) = reshape(x, length(x))
# backward(::BroadcastedOperator{typeof(flatten)}, x, g) = tuple(reshape(g, size(x)))

# recurrence(x::GraphNode, h::GraphNode, w_x::GraphNode, w_h::GraphNode) = let

function tanh_fun(x)
    return (exp.(x) - exp.(-x)) / (exp.(x) + exp.(-x))
end

initial_recurrence(w_i::GraphNode, w_h::GraphNode, i::GraphNode, h::GraphNode, b::GraphNode) =
    let
        # println("INITIAL RECURRENCE")
        return BroadcastedOperator(recurrence, w_i, w_h, i, h, b)
    end
recurrence(w_i::GraphNode, w_h::GraphNode, i::GraphNode, operator::BroadcastedOperator, bias::GraphNode) =
    let
        # println("REPEATED RECURRENCE")
        return BroadcastedOperator(recurrence, w_i, w_h, operator, bias, i) #TODO tutaj jest dziwne to po lewo, chyba pod spodem powinno dzialac madrzej 
        # return BroadcastedOperator(recurrence, w_x, w_h, x, Constant(operator.output))
    end
forward(::BroadcastedOperator{typeof(recurrence)}, w_x, w_h, x, h, b) =
    let

        # if length(w_x) == 12544
            
        # else
        #     res = w_h * h .+ x * w_x
        # end

        # @show size(w_x)
        # @show size(w_h)
        # @show size(x)
        # @show size(h)

        if size(w_x)[2] != size(x)[1]
            # switch values
            buffer = x
            x = b
            b = h
            h = buffer
        end
        
        res = w_x * x .+ w_h * h .+ b
        return res
    end
backward(::BroadcastedOperator{typeof(recurrence)}, w_x, w_h, x, h, b, g) =
    let
        # @show "BACKWARD RECURRENCE"

        if size(h)[1] == 196
            # switch values
            buffer = x
            x = b
            b = h
            h = buffer
        end

        # if size(w_h)[2] != 196
            g_wi = g * x'
            g_wh = g * h'
            g_h = w_h * g
        # else
        #     g_wx = g * h'  #inny recurrence, argumenty są następujące: h -> wx; wx -> wh; wh -> x; x -> h
        #     g_wh = g * w_x'
        #     g_h = g * x'
        # end
        # @show size(tuple(g_wx, g_wh))
        return tuple(g_wi, g_wh, g_h, g)
    end


dense(x::GraphNode, w::GraphNode) = BroadcastedOperator(dense, x, w)
forward(::BroadcastedOperator{typeof(dense)}, x, w) =
    let
        # @show "FORWARD DENSE"
        # @show size(w)
        # @show size(x)
        # @show w
        # println("FORWARD DENSE PRZED: ", x)
        # println("FORWARD DENSE PO: ", w*x)
        return w * x
    end
backward(::BroadcastedOperator{typeof(dense)}, x, w_out, g) =
    let
        # @show "BACKWARD DENSE"
        # @show size(x)
        # @show size(w_out)
        # @show size(g)
        # tuple(w' * g, g * x', g)
        # @show size(g)
        # @show size(x)\
        # @show size(g)
        # println("GRAD W BACKWARD DENSE: ", g)
        # @show size(g)
        # @show typeof(g)
        # @show g
        # @show size(x)
        # @show typeof(x)
        # @show size(w_out)
        # @show typeof(x)
        tuple(w_out' * g, g * x') # g * x' -> gradient dla wag out
    end


tanh(x::GraphNode) = BroadcastedOperator(tanh, x)
forward(::BroadcastedOperator{typeof(tanh)}, x) =
    let
        # @show "TANH FORWARD"
        # @show x
        # println("CONVERTED TO TANH X")
        # @show tanh_fun.(x)
        # @show size(x)

        # println("FORWARD TANH PRZED: ", x)

        # println("FORWARD TANH PO: ", tanh_fun.(x))

        return tanh_fun.(x)
    end
backward(::BroadcastedOperator{typeof(tanh)}, x, g) =
    let
        # @show "BACKWARD TANH"
        # @show size(x)
        # @show size(g)

        # @show size(g)
        # println("GRAD W BACKWARD TANH: ", g)

        # @show g
        # @show size(g)
        # @show size(x)
        dtan = (-(tanh_fun.(x) .^ 2) .+ 1);

        # @show size(dtan)
        # @show size(g)
        return tuple(g .* (-(tanh_fun.(x) .^ 2) .+ 1)) # sprobowac z liczbami dualnymi pozniej
        # end
        # if size(x)[2] != size(g)[1]
        #     return tuple(g' * (-(tanh_fun.(x) .^ 2) .+ 1)) # sprobowac z liczbami dualnymi pozniej
        # else
        #     return tuple((-(tanh_fun.(x) .^ 2) .+ 1) * g) # sprobowac z liczbami dualnymi pozniej
        # end


    end
# relu(x::GraphNode) = BroadcastedOperator(relu, x)
# forward(::BroadcastedOperator{typeof(relu)}, x) = return max.(x, zero(x))
# backward(::BroadcastedOperator{typeof(relu)}, x, g) = return tuple(g .* (x .> 0))

identity_transpose(x::GraphNode) = BroadcastedOperator(identity_transpose, x)
forward(::BroadcastedOperator{typeof(identity_transpose)}, x) = let 
    # println("FORWARD IDENTITY")
    # @show size(x)
    # @show x
    x'    
end


backward(::BroadcastedOperator{typeof(identity_transpose)}, x, g) = let
    # println("BACKWARD IDENTITY")
    # @show size(g)
    # @show size(g)
    # println("GRAD W BACKWARD IDENTITY: ", g)
    # @show size(g)
    # @show g
    # @show g'
    return tuple(g)
end

cross_entropy_loss(y_hat::GraphNode, y::GraphNode) = BroadcastedOperator(cross_entropy_loss, y_hat, y)
forward(::BroadcastedOperator{typeof(cross_entropy_loss)}, y_hat, y) =
    let
		global num_of_clasiffications
        global num_of_correct_clasiffications
        num_of_clasiffications += 1
        y_hat = y_hat'
        
        # println("C_E_L FORWARD PRZED: ", y_hat)
        if argmax(y_hat) == argmax(y)
            num_of_correct_clasiffications += 1
        end
        y_hat = y_hat .- maximum(y_hat)

        # println("C_E_L FORWARD PO: ", y_hat)

        y_hat = exp.(y_hat) ./ sum(exp.(y_hat))
        # @show size(y_hat)
        # @show size(y)
        loss = sum(log.(y_hat) .* y) * -1.0
        return loss
    end
backward(node::BroadcastedOperator{typeof(cross_entropy_loss)}, y_hat, y, g) =
    let

        # @show size(g)
        # println("GRAD W BACKWARD C_E_L (INICJACJA GRADIENTU): ", g)

        y_hat = y_hat'
        y_hat = y_hat .- maximum(y_hat)
        y_hat = exp.(y_hat) ./ sum(exp.(y_hat))

        # @show size(g)
        # @show(y_hat)
        # @show(y)
        # @show((y_hat .- y))
        # println("GRAD W BACKWARD C_E_L (PO): ", g .* (y_hat .- y))
        # @show "BACKWARD C_E_L"
        # @show g .* (y_hat .- y)
        # @show g
        # @show g * (y_hat .- y)
        return tuple(g .* (y_hat .- y))
    end

mse(y_hat::GraphNode, y::GraphNode) = BroadcastedOperator(mse, y_hat, y)
forward(::BroadcastedOperator{typeof(mse)}, y_hat, y) =
    let
        squared_error = 0.0
        for (y_def, y_pred) in zip(y_hat, y)
            squared_error += (y_def - y_pred)^2
        end
        # @show squared_error
        return squared_error
    end
backward(node::BroadcastedOperator{typeof(mse)}, y_hat, y, g) =
    let
        println("PRED: " * string(y_hat))
        res_mse = g .* 2 * (y_hat .- y)
        return tuple(res_mse)
    end


# logsoftmax(y_hat; dims) = y_hat .- log.(sum(exp.(y_hat), dims=dims))
crossentropy(y_hat, y, dims, agg=mean) = agg(-sum(y .* log.(y_hat .+ 3.248761f-10); dims))


# mean(-sum(y .* y_hat .- log.(sum(exp.(y_hat), dims = dims)); dims))

function mean_derivative()

end