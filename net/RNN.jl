
function relu(x::Float64)
    return max(x, zero(x))
end

function tanh(x)
    return (exp.(x) - exp.(-x)) / (exp.(x) + exp.(-x)) # 97.65832
end

function pass!()

end

function rnn_cell(input, hidden_state, weights_i, weights_h)
    hidden_state = tanh.(weights_i * input .+ weights_h * hidden_state)
    return hidden_state, hidden_state
end


include("../diff/Graph.jl")
using Statistics: mean
using Flux: glorot_uniform
# x = rand(Float32, input_size) # dummy input data

# y = [1.555,2.777]


# @show x
# @show h
# @show y


input_size = 196
hidden_output_size = 64
output_size = 10

Wxh = glorot_uniform(hidden_output_size, input_size) # pojedynczy neuron, 196 wejsc, 64 wyjscia -> wagi dla wejsc
# @show Wxh
Whh = glorot_uniform(hidden_output_size, hidden_output_size) # wagi wyjsc (64 wyjscia)
# @show Whh
Why = glorot_uniform(output_size, hidden_output_size)

h = glorot_uniform(hidden_output_size) # random initial hidden state

biases = glorot_uniform(hidden_output_size)

# Wxh = randn(Float32, hidden_output_size, input_size) # pojedynczy neuron, 196 wejsc, 64 wyjscia -> wagi dla wejsc
# # @show Wxh
# Whh = randn(Float32, hidden_output_size, hidden_output_size) # wagi wyjsc (64 wyjscia)
# # @show Whh
# Why = randn(Float32, output_size, hidden_output_size)
# @show Why
# h = rand(Float32, hidden_output_size) # random initial hidden state
@show size(Why)
@show typeof(Why)
# b   = randn(Float32, output_size)

using MLDatasets, Flux
train_data = MLDatasets.MNIST(split=:train)
x1dim = reshape(train_data.features, 28 * 28, :) # reshape 28×28 pixels into a vector of pixels
yhot = Flux.onehotbatch(train_data.targets, 0:9) # make a 10×60000 OneHotMatrix
part_of_train_data = x1dim[:, 1:PART_DATA_AMOUNT]
splitted_train_data = [part_of_train_data[1:196, :], part_of_train_data[197:392, :], part_of_train_data[393:588, :], part_of_train_data[589:end, :]];# 
y_train_part = yhot[:, 1:PART_DATA_AMOUNT]

# println("SIZE OF TEST DATA: " * string(size(x1dim)))

PART_DATA_AMOUNT = 60000;

# @show y
@show size(y_train_part)
@show size(yhot)
"ok"



# println("PART OF TRAIN DATA SIZE: "* string(size(part_of_train_data)))
# println("SPLITTED TRAIN 1: "* string(size(splitted_train_data[1])))
# println("SPLITTED TRAIN 2: "* string(size(splitted_train_data[2])))
# println("SPLITTED TRAIN 3: "* string(size(splitted_train_data[3])))
# println("SPLITTED TRAIN 4: "* string(size(splitted_train_data[4])))

#TODO DO POCIĘCIA NA MNIEJSZE BATCHE (JAK JUZ BEDZIE 60k DANYCH)

start_hidden = randn(Float32, hidden_output_size)


x1_train = Constant(part_of_train_data[1:196, 1])
x2_train = Constant(part_of_train_data[197:392, 1])
x3_train = Constant(part_of_train_data[393:588, 1])
x4_train = Constant(part_of_train_data[589:end, 1])
y_train = Constant(y_train_part[:, 1])
@show size(y_train_part[:, 1])

@show part_of_train_data[1:196, 1]

i_weights = Variable(Wxh, name="ZMIENNE_WEJSCIE")
h_weights = Variable(Whh, name="ZMIENNE_UKRYTE")
o_weights = Variable(Why, name="ZMIENNE_OUT")
b = Variable(biases, name="BIASES")

graph = build_graph(x1_train, x2_train, x3_train, x4_train, start_hidden, y_train, i_weights, h_weights, o_weights, b)
# last_order_output = last(graph).output 


for x in graph
    @show typeof(x)
end

println("Przed forward")
#@show graph
#@show last_order_output

#TODO jest popsute, siadz do tego na spokojnie...

# IT_NUM = 20
BATCH_SIZE = 100
EPOCHS_NUM = 5
LR = 15e-3

@time for j in 1:EPOCHS_NUM
    global num_of_clasiffications = 0
    global num_of_correct_clasiffications = 0
    for i in 1:PART_DATA_AMOUNT
        blad_petla = forward!(graph)
        backward!(graph)
        if i % BATCH_SIZE == 0
            # println("Funkcja starty: ", blad_petla)
            update_weights!(graph, LR, BATCH_SIZE)
        end
        x1_train.output = part_of_train_data[1:196, i]
        x2_train.output = part_of_train_data[197:392, i]
        x3_train.output = part_of_train_data[393:588, i]
        x4_train.output = part_of_train_data[589:end, i]
        y_train.output = y_train_part[:, i]
    end
    blad = forward!(graph)
    println("Funkcja starty: ", blad)
    println("[TRAIN] klasyfikacje poprawne: ", num_of_correct_clasiffications)
    println("[TRAIN] klasyfikacje: ", num_of_clasiffications)
    println("train acc: ", num_of_correct_clasiffications / num_of_clasiffications)
end


test_data = MLDatasets.MNIST(split=:test)
x1dim_test = reshape(test_data.features, 28*28, :)
PART_DATA_TEST_AMOUNT = 10000;
yhot_test = Flux.onehotbatch(test_data.targets, 0:9)
part_of_test_data = x1dim_test[:, 1:PART_DATA_TEST_AMOUNT]
splitted_test_data = [part_of_test_data[1:196, :], part_of_test_data[197:392, :], part_of_test_data[393:588, :], part_of_test_data[589:end, :]];# 
y_test_part = yhot_test[:, 1:PART_DATA_TEST_AMOUNT]

x1_test = Constant(part_of_test_data[1:196, 1])
x2_test = Constant(part_of_test_data[197:392, 1])
x3_test = Constant(part_of_test_data[393:588, 1])
x4_test = Constant(part_of_test_data[589:end, 1])
y_test = Constant(y_test_part[:, 1])

hidden_init_state = glorot_uniform(hidden_output_size) # random initial hidden state

test_graph = build_graph(x1_test, x2_test, x3_test, x4_test, hidden_init_state, y_test, i_weights, h_weights, o_weights, b)

global num_of_correct_clasiffications = 0
global num_of_clasiffications = 0

for i in 1:PART_DATA_TEST_AMOUNT
    loss_test = forward!(test_graph)
    # backward!(graph)
    # if i % BATCH_SIZE == 0
    #     println("Funkcja starty: ", blad_petla)
    #     update_weights!(graph, 0.1, BATCH_SIZE)
    # end
    x1_test.output = part_of_test_data[1:196, i]
    x2_test.output = part_of_test_data[197:392, i]
    x3_test.output = part_of_test_data[393:588, i]
    x4_test.output = part_of_test_data[589:end, i]
    y_test.output = y_test_part[:, i]
end

println("[TEST] klasyfikacje poprawne: ", num_of_correct_clasiffications)
println("[TEST] klasyfikacje: ", num_of_clasiffications)
println("test acc: ", num_of_correct_clasiffications / num_of_clasiffications)