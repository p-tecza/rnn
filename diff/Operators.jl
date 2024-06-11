

# flatten(x::GraphNode) = BroadcastedOperator(flatten, x)
# forward(::BroadcastedOperator{typeof(flatten)}, x) = reshape(x, length(x))
# backward(::BroadcastedOperator{typeof(flatten)}, x, g) = tuple(reshape(g, size(x)))

# recurrence(x::GraphNode, h::GraphNode, w_x::GraphNode, w_h::GraphNode) = let

function tanh_fun(x)
    return (exp.(x) - exp.(-x)) / (exp.(x) + exp.(-x))
end

function softmax_fun(x)
    # res = zeros(size(x))
    # println("SOFTMAXXX")
    # @show size(x)
    # @show typeof(x)
    # @show size(res)
    # @show typeof(res)
    # for i in 1:size(x)[1]
        # observation = x[i, :]
        # @show observation
        
        # res[i, :] = exp.(x) ./ col_sum
        # @show res
    # end
    # @show size(res)
    # @show size(x)
    # @show x
    col_sum = sum(exp.(x))
    res = exp.(x) ./ col_sum
    # @show size(res)
    # @show res
    return res
end

function softmax_jacob_matrix(s)
    # ZROB TO ZEBY DZIALALO NA POJEDYNCZEJ OBSERWACJI
    # res = []
    # @show size(s)
    # for i in 1:size(s)[1]
        # observation = s[i, :]
        # @show size(observation)
        # @show length(observation)
        # @show observation
        collection_length = size(s)[1]
        observation_jacobian = zeros(collection_length, collection_length)
        # @show size(observation_jacobian)
        for j in 1:collection_length
            for k in 1:collection_length
                if j == k
                    observation_jacobian[j,k] = s[j] * (1 - s[j])
                    continue
                end
                observation_jacobian[j,k] = -s[j] * s[k]
            end
        end
        # @show size(observation_jacobian)
        # # res[i,:] = observation_jacobian
        # push!(res, observation_jacobian)
    # end
    @show size(observation_jacobian)
    return observation_jacobian
end

initial_recurrence(w_x::GraphNode, w_h::GraphNode, x::GraphNode, h::GraphNode) =
    let
        # println("INITIAL RECURRENCE")
        return BroadcastedOperator(recurrence, w_x, w_h, x, h)
    end
forward(::BroadcastedOperator{typeof(recurrence)}, w_x, w_h, x, h) =
    let

        # NAPRAWIANIE GRAFU
        # ind = length(n.inputs) == 3 ? 2 : 1
        # n.inputs = n.inputs[ind:length(n.inputs)]

        # println("cojestwn [n]: ",n)
        # println("cojestwn: [n.inputs]: ",n.inputs)
        # println("cojestwn: [n.output]: ",n.output)
        # println("cojestwn: [n.gradient]: ",n.gradient)
        # println("cojestwn: [n.cache]: ",n.cache)

        # @show "FORWARD RECURRENCE"
        # @show size(w_x)
        # @show size(x)
        # @show size(w_h)
        # @show size(h)
        if size(w_x)[2] == size(x)[1]
            res = w_x * x .+ w_h * h
        else
            res = w_h * h .+ x * w_x
        end

        # @show size(res)
        return res
    end
backward(::BroadcastedOperator{typeof(recurrence)}, w_x, w_h, x, h, g) =
    let
        # @show "BACKWARD RECURRENCE"
        # @show size(w_x)
        # @show size(x)
        # @show size(w_h)
        # @show size(h)
        # @show size(g)
        if size(g)[2] == size(x')[1]
            # println("BACKWARD RECURRENCE PRZED INPUTEM")
            # println("WYMIARY X: ",size(x))
            # println("WYMIARY H: ",size(h))
            g_wx = g * x'
            # g_wh = g .* h
            g_wh = g * h'
        else
            # println("BACKWARD RECURRENCE W REKURENCJI")
            # println("WYMIARY X: ",size(h))
            # println("WYMIARY H: ",size(w_x))
            g_wx = g * h'  #inny recurrence, argumenty są następujące: h -> wx; wx -> wh; wh -> x; x -> h
            # g_wh = g .* w_x 
            g_wh = g * w_x'
        end

        # @show size(g_wh)
        # @show size(g_wx)
        # println("KONIEC BACKWARD RECURRENCE")
        return tuple(g_wx, g_wh)
    end

recurrence(operator::BroadcastedOperator, w_x::GraphNode, w_h::GraphNode, x::GraphNode) =
    let
        # println("REPEATED RECURRENCE")
        return BroadcastedOperator(recurrence, operator, w_x, w_h, x) #TODO tutaj jest dziwne to po lewo, chyba pod spodem powinno dzialac madrzej 
        # return BroadcastedOperator(recurrence, w_x, w_h, x, Constant(operator.output))
    end


dense(x::GraphNode, w::GraphNode) = BroadcastedOperator(dense, x, w)
forward(::BroadcastedOperator{typeof(dense)}, x, w) =
    let
        # @show "FORWARD DENSE"
        # @show size(w)
        # @show size(x)
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
        # @show size(x)
        tuple(1.0, (x * g)') # g * x' -> gradient dla wag out
    end


tanh(x::GraphNode) = BroadcastedOperator(tanh, x)
forward(::BroadcastedOperator{typeof(tanh)}, x) =
    let
        # @show "TANH FORWARD"
        # @show x
        # println("CONVERTED TO TANH X")
        # @show tanh_fun.(x)
        # @show size(x)
        return tanh_fun.(x)
    end
backward(::BroadcastedOperator{typeof(tanh)}, x, g) =
    let
        # @show "BACKWARD TANH"
        # @show size(x)
        # @show size(g)

        dtan = (-(tanh_fun.(x) .^ 2) .+ 1);

        # @show size(dtan)
        # @show size(g)
        return tuple(g * (-(tanh_fun.(x) .^ 2) .+ 1)) # sprobowac z liczbami dualnymi pozniej
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
    x'    
end


backward(::BroadcastedOperator{typeof(identity_transpose)}, x, g) = let
    # println("BACKWARD IDENTITY")
    # @show size(g)
    tuple(g')
end

# softmax(x::GraphNode) = BroadcastedOperator(softmax, x)
# forward(::BroadcastedOperator{typeof(softmax)}, x) = let
#     @show "FORWARD SOFTMAX"
#     @show size(x)
#     @show size(x[:,1])

#     cosik = broadcast(softmax_fun, x[:,i] for i in 1:size(x)[2])
#     @show size(cosik)
#     @show length(cosik)
#     # @show cosik
#     rearranged = mapreduce(permutedims, vcat, cosik)
#     @show size(rearranged)
#     return rearranged
# end
# backward(::BroadcastedOperator{typeof(softmax)}, x, g) = let
#     @show "BACKWARD SOFTMAX"
#     @show size(x)
#     s = broadcast(softmax_fun, x[:,i] for i in 1:size(x)[2])
#     s = mapreduce(permutedims, vcat, s)
#     @show size(s)
#     jac = broadcast(softmax_jacob_matrix, s[i,:] for i in size(s)[1])
#     @show size(g)
#     @show size(jac)
#     return g * jac
# end

# cross_entropy_loss(y_hat::GraphNode, y::GraphNode) = BroadcastedOperator(cross_entropy_loss, y_hat, y)
# forward(::BroadcastedOperator{typeof(cross_entropy_loss)}, y_hat, y) =
#     let
#         #TODO jakos dziwnie to jest liczone
#         @show "FORWARD CROSS_ENTROPY"
#         # @show size(y_hat)
#         # @show size(y)
#         num_of_clasiffications = 0 # tu bylo global bez wartosci
#         num_of_correct_clasiffications = 0 # tu bylo global bez wartosci
#         num_of_clasiffications += 1
#         if argmax(y_hat) == argmax(y)
#             num_of_correct_clasiffications += 1
#         end

#         # @show y'
#         # @show y_hat'
#         y_hat = y_hat'

#         @show size(y)
#         @show size(y')
#         @show size(y_hat)

#         # println("SOFTMAX OD Y")
#         # softmax_fun(y_hat)
#         # println("SOFTMAX OD Y_HAT")
#         # smax = softmax_fun(y_hat')

#         # smax_jac = softmax_jacob_matrix(smax)

#         # mean_in_y_hat = mean(y_hat)

#         # y_hat = y_hat.()

#         # y_hat = y_hat .- maximum(y_hat)
#         # y_hat = exp.(y_hat) ./ sum(exp.(y_hat))
#         # loss = sum(log.(y_hat) .* y) * -1.0

#         # return loss
#         return crossentropy(y_hat, y, size(y_hat))
#     end
# backward(node::BroadcastedOperator{typeof(cross_entropy_loss)}, y_hat, y, g) =
#     let
#         @show "BACKWARD CROSS_ENTROPY"
#         # y_hat = y_hat .- maximum(y_hat)
#         # y_hat = exp.(y_hat) ./ sum(exp.(y_hat))
#         # @show g
#         # @show tuple(g .* (y_hat .- y))
#         # mean_d = similar(y_hat)
#         mean_d = fill(1 / size(y_hat)[2], size(y_hat))
#         @show size(y_hat)
#         @show g
#         @show size(g)
#         @show tuple(g .* mean_d)
#         return tuple(g .* mean_d)
#     end

cross_entropy_loss(y_hat::GraphNode, y::GraphNode) = BroadcastedOperator(cross_entropy_loss, y_hat, y)
forward(::BroadcastedOperator{typeof(cross_entropy_loss)}, y_hat, y) =
    let
		num_of_clasiffications = 0
        num_of_correct_clasiffications = 0
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


# logsoftmax(y_hat; dims) = y_hat .- log.(sum(exp.(y_hat), dims=dims))
crossentropy(y_hat, y, dims, agg=mean) = agg(-sum(y .* log.(y_hat .+ 3.248761f-10); dims))


# mean(-sum(y .* y_hat .- log.(sum(exp.(y_hat), dims = dims)); dims))

function mean_derivative()

end