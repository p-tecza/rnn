

import Base: show, summary

abstract type GraphNode end
abstract type Operator <: GraphNode end

mutable struct Constant <: GraphNode
    output::Any
end

mutable struct Variable <: GraphNode
    output::Any
    gradient::Any
    name::String
    batch_gradient::Any
    recurrent_procedure::Bool
    Variable(output; name="?") = new(output, nothing, name, nothing, true)
end

mutable struct BroadcastedOperator{F} <: Operator
    inputs::Any
    output::Any
    gradient::Any
    cache::Any
    function BroadcastedOperator(fun, inputs...)
        return new{typeof(fun)}(inputs, nothing, nothing, nothing)
    end
end

function visit(node::GraphNode, visited, order)
    if node ∉ visited
        push!(visited, node)
        push!(order, node)
    end
    return nothing
end

function visit(node::Operator, visited, order)
    if node ∉ visited
        push!(visited, node)
        for input in node.inputs
            # if isa(node, BroadcastedOperator{typeof(recurrence)}) && isa(input, BroadcastedOperator{typeof(tanh)}) 
            visit(input, visited, order)
        end
        push!(order, node)
    end
    return nothing
end

function topological_sort(head::GraphNode)
    visited = Set()
    order = Vector()
    visit(head, visited, order)
    return order
end

import Base: show, summary
# show(io::IO, x::ScalarOperator{F}) where {F} = print(io, "op ", x.name, "(", F, ")");
show(io::IO, x::BroadcastedOperator{F}) where {F} = print(io, "op.", "BR_OP", "(", F, ")");
show(io::IO, x::Constant) = print(io, "const ", x.output)
show(io::IO, x::Variable) = begin
    print(io, "var ", x.name)
    print(io, "\n ┣━ ^ ")
    summary(io, x.output)
    print(io, "\n ┗━ ∇ ")
    summary(io, x.gradient)
end


include("Operators.jl")

function build_graph(input1, input2, input3, input4, start_hidden, output, i_weights, h_weights, out_weights, biases)

    known_output = output
    # i_weights = Variable(i_weights, name="ZMIENNE_WEJSCIE")
    # h_weights = Variable(h_weights, name="ZMIENNE_UKRYTE")
    h = Constant(start_hidden)
    recurrence_started = false
    # @show size(full_input)
    res1 = initial_recurrence(i_weights, h_weights, input1, h, biases) |> tanh # to tanh raczej nie tu
    res2 = recurrence(i_weights, h_weights, input2, res1, biases) |> tanh
    res3 = recurrence(i_weights, h_weights, input3, res2, biases) |> tanh
    res = recurrence(i_weights, h_weights, input4, res3, biases) |> tanh
    l_dense = dense(res, out_weights) |> identity_transpose
    e = cross_entropy_loss(l_dense, known_output)

    return topological_sort(e)
end

function update_weights!(graph::Vector, lr::Float64, batch_size::Int64)
    for node in graph
        if isa(node, Variable) && hasproperty(node, :batch_gradient)
            # if length(node.batch_gradient) == 1
            #     node.batch_gradient /= batch_size
            # else

            # end

            # println(node.name,".output:")

            node.batch_gradient ./= batch_size

            # if length(node.output) == 1
            #     node.output -= lr * node.batch_gradient 
            # else
            #     node.output .-= lr * node.batch_gradient 
            # end
            # @show size(node.output)
            # @show size(node.batch_gradient)
            # @show node.name
            # println("UPDATE WEIGHTS WAGI PRZED: ",node.output)
            node.output .-= lr * node.batch_gradient
            # weights_to_rescale = node.output
            # limited_weight = min.(weights_to_rescale, 1)
            # limited_weight = max.(weights_to_rescale, -1)
            # node.output = weights_to_rescale
            # node.output = 2 .* ((node.output .- minimum(weights_to_rescale)) ./ (maximum(weights_to_rescale) - minimum(weights_to_rescale))) .- 1
            
            if isa(node.output, Matrix)


                # println("maxweight: ", maximum(node.output))
                # println("minweight: ", minimum(node.output))

                # println("WAGA ",node.name,":",node.output[1:10])
            elseif node.name == "BIASES"
                # println("BIASES: ",node.output[1:10])
            end


            # println("UPDATE WEIGHTS WAGI PO: ",node.output)
            node.batch_gradient = nothing
            # fill(node.batch_gradient, nothing)
        end
    end
end

# @show 5.0

function forward!(order::Vector)
    for node in order
        compute!(node)
        reset!(node)
    end
    return last(order).output
end

reset!(node::Constant) = nothing
reset!(node::Variable) = let
    node.gradient = nothing
    node.recurrent_procedure = true
end
reset!(node::Operator) = node.gradient = nothing

compute!(node::Constant) = nothing
compute!(node::Variable) = nothing
compute!(node::Operator) = node.output = forward(node, [input.output for input in node.inputs]...)

update!(node::Constant, gradient, final_input_gradient::String) =
    let
        # println("UPDATE! = NOTHING??????")
        nothing
    end
update!(node::GraphNode, gradient, final_input_gradient::String) =
    let
        # @show gradient
        # for i in eachindex(gradient)
        #     if gradient[i] < -5.
        #         gradient[i] = -5.
        #     elseif gradient[i] > 5.
        #         gradient[i] = 5.
        #     end
        # end
        
        # println("ROZMIAR GRADIENTU: ", size(gradient))
        # println("TYP GRADIENTU: ", typeof(gradient))
        # if typeof(gradient) == Matrix{Float64}
        #     # @show gradient[1:7,1:7]
        #     limited_grad = min.(gradient, 5)
        #     limited_grad = max.(limited_grad, -5)
        #     # @show limited_grad[1:7,1:7]
        #     gradient = limited_grad
        # end

        node.gradient = gradient


        # if isa(node,Variable) && node.name == "BIASES"
        #     @show node.gradient
        # end

        if typeof(node) == Variable
            if final_input_gradient == "HIDDEN" && (node.name == "ZMIENNE_WEJSCIE" || node.name == "BIASES")
                if isnothing(node.batch_gradient)
                    node.batch_gradient = gradient
                else
                    node.batch_gradient .+= gradient
                end
            elseif final_input_gradient == "HIDDEN" && node.name == "ZMIENNE_UKRYTE"
                if isnothing(node.batch_gradient)
                    # println("[B] INICJUJE BATCH GRADIENT W VARZE: " * node.name)
                    # println("PRZYPISUJE HIDDEN")
                    node.batch_gradient = gradient
                else
                    # println(node.name,":",node)
                    # println("[B] DODAJE GRADIENT DO BATCH GRADIENT W VARZE: ", node.name)
                    # println(node.name)
                    # println(size(gradient))
                    # println(size(node.batch_gradient))
                    # println("DODAJE HIDDEN")
                    node.batch_gradient .+= gradient
                end
            elseif node.name == "ZMIENNE_OUT"
                if isnothing(node.batch_gradient)
                    # println("PRZYPISUJE OUT")
                    # println("[B] INICJUJE BATCH GRADIENT W VARZE: " * node.name)
                    node.batch_gradient = gradient
                else
                    # println("[B] DODAJE GRADIENT DO BATCH GRADIENT W VARZE: ", node.name)
                    # println(node.name)
                    # println(size(gradient))
                    # println(size(node.batch_gradient))
                    # println("DODAJE OUT")
                    node.batch_gradient .+= gradient
                end
            end
        end
    end

function backward!(node::Constant, final_input_gradient::String) end
function backward!(node::Variable, final_input_gradient::String) end
function backward!(order::Vector; seed=1.0)
    result = last(order)
    result.gradient = seed
    @assert length(result.output) == 1 "Gradient is defined only for scalar functions"
    for (index, node) in enumerate(reverse(order))
        # println("PODROZ W TYL: ", index, node)
        if index == 16 # 16 to index ostatniej pierwszej warstwy recurrence, z niej bierzemy gradient DLA HIDDEN (a dla INPUT i BIASES index 7)
            # println("INDEX: ",index," NODE: ",node)
            backward!(node, "")
        elseif index == 7
            backward!(node, "HIDDEN") # "INPUT_AND_BIASES"
        else
            backward!(node, "")
        end
        if isa(node, BroadcastedOperator) # obcinanie gradientu
            gradient = node.gradient
            if typeof(gradient) == Matrix{Float64}
                # # @show gradient[1:7,1:7]
                # limited_grad = min.(gradient, 5)
                # limited_grad = max.(limited_grad, -5)
                # # @show limited_grad[1:7,1:7]
                # node.gradient = limited_grad
            else

            end
        
        end
        
    end
    return nothing
end

function backward!(node::Operator, final_input_gradient::String)
    inputs = node.inputs
    # println("INPUTS: ")
    # for i in inputs
    #     @show typeof(i)
    #     @show size(i.output)
    # end

    # @show size(node.gradient)

    gradients = backward(node, [input.output for input in inputs]..., node.gradient)
    #@show inputs
    #@show gradients
    # println("START UPDATE!")
    # @show inputs
    # @show gradients
    for (input, gradient) in zip(inputs, gradients)

        # @show size(gradient)
        # @show size(gradient)
        # @show input
        # if length(gradients) > 1 && isa(node, BroadcastedOperator{typeof(recurrence)}) && isa(input, BroadcastedOperator{typeof(tanh)})
        #     gradient = gradients[2]
        # elseif length(gradients) > 1 && isa(node, BroadcastedOperator{typeof(recurrence)}) && isa(input, Variable) && input.name == "ZMIENNE_WEJSCIE"
        #     gradient = gradients[1]
        # end
        update!(input, gradient, final_input_gradient)
    end
    # println("KONIEC UPDATE!")
    return nothing
end