include("../diff/Graph.jl")
using Flux: glorot_uniform

input_size = 196
hidden_output_size = 64
output_size = 10

Wxh = glorot_uniform(hidden_output_size, input_size)
Whh = glorot_uniform(hidden_output_size, hidden_output_size)
Why = glorot_uniform(output_size, hidden_output_size)
start_hidden = zeros(hidden_output_size)
biases = zeros(hidden_output_size)

using MLDatasets, Flux

PART_DATA_AMOUNT = 60000;
train_data = MLDatasets.MNIST(split=:train)
x1dim = reshape(train_data.features, 28 * 28, :)
yhot = Flux.onehotbatch(train_data.targets, 0:9)
part_of_train_data = x1dim[:, 1:PART_DATA_AMOUNT]
splitted_train_data = [part_of_train_data[1:196, :], part_of_train_data[197:392, :], part_of_train_data[393:588, :], part_of_train_data[589:end, :]];# 
y_train_part = yhot[:, 1:PART_DATA_AMOUNT]

x1_train = Constant(part_of_train_data[1:196, 1])
x2_train = Constant(part_of_train_data[197:392, 1])
x3_train = Constant(part_of_train_data[393:588, 1])
x4_train = Constant(part_of_train_data[589:end, 1])
y_train = Constant(y_train_part[:, 1])

i_weights = Variable(Wxh, name="ZMIENNE_WEJSCIE")
h_weights = Variable(Whh, name="ZMIENNE_UKRYTE")
o_weights = Variable(Why, name="ZMIENNE_OUT")
b = Variable(biases, name="BIASES")

graph = build_graph(x1_train, x2_train, x3_train, x4_train, start_hidden, y_train, i_weights, h_weights, o_weights, b)

BATCH_SIZE = 100
EPOCHS_NUM = 5
LR = 15e-3

println("Training started...")
@time for j in 1:EPOCHS_NUM
    global num_of_clasiffications = 0
    global num_of_correct_clasiffications = 0
    for i in 1:PART_DATA_AMOUNT
        blad_petla = forward!(graph)
        backward!(graph)
        if i % BATCH_SIZE == 0
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

test_graph = build_graph(x1_test, x2_test, x3_test, x4_test, start_hidden, y_test, i_weights, h_weights, o_weights, b)

global num_of_correct_clasiffications = 0
global num_of_clasiffications = 0

for i in 1:PART_DATA_TEST_AMOUNT
    loss_test = forward!(test_graph)
    x1_test.output = part_of_test_data[1:196, i]
    x2_test.output = part_of_test_data[197:392, i]
    x3_test.output = part_of_test_data[393:588, i]
    x4_test.output = part_of_test_data[589:end, i]
    y_test.output = y_test_part[:, i]
end

println("[TEST] klasyfikacje poprawne: ", num_of_correct_clasiffications)
println("[TEST] klasyfikacje: ", num_of_clasiffications)
println("test acc: ", num_of_correct_clasiffications / num_of_clasiffications)