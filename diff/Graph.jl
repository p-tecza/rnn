

import Base: show, summary

abstract type GraphNode end
abstract type Operator <: GraphNode end

mutable struct Constant <: GraphNode
    output :: Any
end

mutable struct Variable <: GraphNode
    output :: Any
    gradient :: Any
    name::String
    batch_gradient::Any
    Variable(output; name = "?") = new(output, nothing, name, nothing)
end

mutable struct BroadcastedOperator{F} <: Operator
    inputs :: Any
    output :: Any
    gradient :: Any
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
    print(io, "var ", x.name);
    print(io, "\n ┣━ ^ "); summary(io, x.output)
    print(io, "\n ┗━ ∇ ");  summary(io, x.gradient)
end


include("Operators.jl")

function build_graph(full_input, start_hidden, known_output, i_weights, h_weights, out_weights)
    layer_input = full_input; #[:,1]
    layer_input = Constant(layer_input)
    known_output = Constant(known_output)
    i_weights = Variable(i_weights)
    h_weights = Variable(h_weights)
    h = Constant(start_hidden)
    
    res = recurrence(i_weights, h_weights, layer_input, h) # |> tanh # to tanh raczej nie tu 


    out_weights = Variable(out_weights) #tutaj też optymalizacja potrzebna
    #res = dense(res, out_weights) |> identity TODO ODKOMENTUJ
    e = mse(res, known_output)

	return topological_sort(e)
end

function update_weights!(graph::Vector, lr::Float64, batch_size::Int64)
    for node in graph
        if isa(node, Variable) && hasproperty(node, :batch_gradient)
            # if length(node.batch_gradient) == 1
            #     node.batch_gradient /= batch_size
            # else
                
            # end
            node.batch_gradient ./= batch_size

            # if length(node.output) == 1
            #     node.output -= lr * node.batch_gradient 
            # else
            #     node.output .-= lr * node.batch_gradient 
            # end

            node.output .-= lr * node.batch_gradient 
            fill(node.batch_gradient, 0)
        end
    end
end

@show 5.0

function forward!(order::Vector)
    for node in order
        compute!(node)
        reset!(node)
    end
    return last(order).output
end

reset!(node::Constant) = nothing
reset!(node::Variable) = node.gradient = nothing
reset!(node::Operator) = node.gradient = nothing

compute!(node::Constant) = nothing
compute!(node::Variable) = nothing
compute!(node::Operator) = node.output = forward(node, [input.output for input in node.inputs]...)

update!(node::Constant, gradient) = let
    println("UPDATE! = NOTHING??????")
    nothing
end
update!(node::GraphNode, gradient) = let
    # @show gradient
    # for i in eachindex(gradient)
    #     if gradient[i] < -5.
    #         gradient[i] = -5.
    #     elseif gradient[i] > 5.
    #         gradient[i] = 5.
    #     end
    # end
    # @show gradient
    node.gradient = gradient
    #@show node.gradient
    #@show typeof(node.gradient)
    if typeof(node) == Variable
        if isnothing(node.batch_gradient)
            node.batch_gradient = gradient
        else
            #@show node.batch_gradient
            #@show typeof(node.batch_gradient)
            #@show length(node.batch_gradient)
            node.batch_gradient .+= gradient
        end
    end
end

function backward!(node::Constant) end
function backward!(node::Variable) end
function backward!(order::Vector; seed=1.0)
    result = last(order)
    result.gradient = seed
    @assert length(result.output) == 1 "Gradient is defined only for scalar functions"
    for node in reverse(order)
        backward!(node)
    end
    return nothing
end

function backward!(node::Operator)
    inputs = node.inputs
    
    gradients = backward(node, [input.output for input in inputs]..., node.gradient)
    #@show inputs
    #@show gradients
    println("START UPDATE!")
    @show inputs
    @show gradients
    for (input, gradient) in zip(inputs, gradients)
        println("INPUT + GRADIENT")
        @show input
        @show gradient
        update!(input, gradient)
    end
    println("KONIEC UPDATE!")
    return nothing
end