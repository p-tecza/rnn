
function relu(x::Float64)
return max(x,zero(x))
end

function tanh(x)
    return (exp.(x) - exp.(-x))/(exp.(x)+exp.(-x))
end

function pass!()

end

function rnn_cell(input, hidden_state, weights_i, weights_h)
    hidden_state = tanh.(weights_i * input .+ weights_h * hidden_state)
    return hidden_state, hidden_state
end


include("../diff/Graph.jl")
using Statistics: mean
# x = rand(Float32, input_size) # dummy input data

# y = [1.555,2.777]


# @show x
# @show h
# @show y


input_size = 196
hidden_output_size = 64
output_size = 10

Wxh = randn(Float32, hidden_output_size, input_size) # pojedynczy neuron, 196 wejsc, 64 wyjscia -> wagi dla wejsc
# @show Wxh
Whh = randn(Float32, hidden_output_size, hidden_output_size) # wagi wyjsc (64 wyjscia)
# @show Whh
Why = randn(Float32, output_size, hidden_output_size)
# @show Why
h = rand(Float32, hidden_output_size) # random initial hidden state
@show size(Why)
@show typeof(Why)
# b   = randn(Float32, output_size)

using MLDatasets, Flux
train_data = MLDatasets.MNIST(split=:train)
test_data  = MLDatasets.MNIST(split=:test)
x1dim = reshape(train_data.features, 28 * 28, :) # reshape 28×28 pixels into a vector of pixels
yhot  = Flux.onehotbatch(train_data.targets, 0:9) # make a 10×60000 OneHotMatrix
println("SIZE OF TEST DATA: "* string(size(x1dim)))

PART_DATA_AMOUNT = 1;

part_of_train_data = x1dim[:,1:PART_DATA_AMOUNT]
splitted_train_data = [part_of_train_data[1:196,:],part_of_train_data[197:392,:],part_of_train_data[393:588,:],part_of_train_data[589:end,:]];
y = yhot[:,1:PART_DATA_AMOUNT]
@show size(y)
@show size(yhot)
"ok"

println("PART OF TRAIN DATA SIZE: "* string(size(part_of_train_data)))
println("SPLITTED TRAIN 1: "* string(size(splitted_train_data[1])))
println("SPLITTED TRAIN 2: "* string(size(splitted_train_data[2])))
println("SPLITTED TRAIN 3: "* string(size(splitted_train_data[3])))
println("SPLITTED TRAIN 4: "* string(size(splitted_train_data[4])))

#TODO DO POCIĘCIA NA MNIEJSZE BATCHE (JAK JUZ BEDZIE 60k DANYCH)

start_hidden = randn(Float32, hidden_output_size, PART_DATA_AMOUNT)

graph = build_graph(splitted_train_data, start_hidden, y, Wxh, Whh, Why)
# last_order_output = last(graph).output 


for x in graph
    @show typeof(x)
end

println("Przed forward")
#@show graph
#@show last_order_output

#TODO jest popsute, siadz do tego na spokojnie...

IT_NUM = 1
BATCH_SIZE = 5

for i in 1:IT_NUM
    blad_petla = forward!(graph)
    @show blad_petla
    backward!(graph)
    update_weights!(graph, 0.1, IT_NUM) # IT_NUM do zmiany pozniej na batch size
end
blad = forward!(graph)