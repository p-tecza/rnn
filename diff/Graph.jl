

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
    Variable(output; name="?") = new(output, nothing, name, nothing)
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
    h = Constant(start_hidden)
    res1 = initial_recurrence(i_weights, h_weights, input1, h, biases) |> tanh
    res2 = recurrence(i_weights, h_weights, input2, res1, biases) |> tanh
    res3 = recurrence(i_weights, h_weights, input3, res2, biases) |> tanh
    res = recurrence(i_weights, h_weights, input4, res3, biases) |> tanh
    l_dense = dense(res, out_weights) |> identity_transpose
    e = cross_entropy_loss(l_dense, output)
    return topological_sort(e)
end

function update_weights!(graph::Vector, lr::Float64, batch_size::Int64)
    for node in graph
        if isa(node, Variable) && hasproperty(node, :batch_gradient)
            node.batch_gradient ./= batch_size
            node.output .-= lr * node.batch_gradient
            fill(node.batch_gradient, 0)
        end
    end
end

function forward!(order::Vector)
    for node in order
        compute!(node)
        reset!(node)
    end
    return last(order).output
end

reset!(node::Constant) = nothing
reset!(node::Variable) =
    let
        node.gradient = nothing
    end
reset!(node::Operator) = node.gradient = nothing

compute!(node::Constant) = nothing
compute!(node::Variable) = nothing
compute!(node::Operator) = node.output = forward(node, [input.output for input in node.inputs]...)

update!(node::Constant, gradient, pick_gradient::Bool) =
    let
        nothing
    end
update!(node::GraphNode, gradient, pick_gradient::Bool) =
    let
        node.gradient = gradient
        if typeof(node) == Variable
            if pick_gradient && node.name != "ZMIENNE_OUT"
                if isnothing(node.batch_gradient)
                    node.batch_gradient = gradient
                else
                    node.batch_gradient .+= gradient
                end
            elseif node.name == "ZMIENNE_OUT"
                if isnothing(node.batch_gradient)
                    node.batch_gradient = gradient
                else
                    node.batch_gradient .+= gradient
                end
            end
        end
    end

function backward!(node::Constant, pick_gradient::Bool) end
function backward!(node::Variable, pick_gradient::Bool) end
function backward!(order::Vector; seed=1.0)
    result = last(order)
    result.gradient = seed
    @assert length(result.output) == 1 "Gradient is defined only for scalar functions"
    for (index, node) in enumerate(reverse(order))
        if index == 7 # gradienty z 7. node są kluczowe do nauki
            backward!(node, true)
        else
            backward!(node, false)
        end
    end
    return nothing
end

function backward!(node::Operator, pick_gradient::Bool)
    inputs = node.inputs
    gradients = backward(node, [input.output for input in inputs]..., node.gradient)
    for (input, gradient) in zip(inputs, gradients)
        update!(input, gradient, pick_gradient)
    end
    return nothing
end