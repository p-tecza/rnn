
function relu(x::Float64)
return max(x,zero(x))
end

function tanh(x)
    return (exp.(x) - exp.(-x))/(exp.(x)+exp.(-x))
end

function pass!()

end

output_size = 2
input_size = 3
Wxh = randn(Float32, output_size, input_size) # pojedynczy neuron, 196 wejsc, 64 wyjscia -> wagi dla wejsc
@show Wxh
Whh = randn(Float32, output_size, output_size) # wagi wyjsc (64 wyjscia)
@show Whh
Why = randn(Float32, output_size, output_size)
@show Why

# b   = randn(Float32, output_size)

function rnn_cell(input, hidden_state, weights_i, weights_h)
    hidden_state = tanh.(weights_i * input .+ weights_h * hidden_state)
    return hidden_state, hidden_state
end


include("../diff/Graph.jl")

x = rand(Float32, input_size) # dummy input data
h = rand(Float32, output_size) # random initial hidden state
y = [1.555,2.777]


@show x
@show h
@show y


start_hidden = similar(y)

graph = build_graph(x, start_hidden, y, Wxh, Whh, Why)
last_order_output = last(graph).output 


for x in graph
    @show typeof(x)
end

println("Przed forward")
#@show graph
#@show last_order_output

for i in 1:100
    blad_petla = forward!(graph)
    backward!(graph)
    update_weights!(graph, 0.1, input_size)
end
blad = forward!(graph)

