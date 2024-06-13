include("../diff/Graph.jl")
using Flux: glorot_uniform

mutable struct MyRNNetwork
    i_weights::Variable
    h_weights::Variable
    o_weights::Variable
    biases::Variable
    start_hidden::Vector{Float64}
    i_size::Int32
    h_size::Int32
    o_size::Int32
end

function init_net(input_size::Int64, hidden_size::Int64, output_size::Int64)
    println("Initializing network...")
    i_size = input_size
    h_size = hidden_size
    o_size = output_size
    Wxh = glorot_uniform(h_size, i_size)
    Whh = glorot_uniform(h_size, h_size)
    Why = glorot_uniform(o_size, h_size)
    start_hidden = zeros(h_size)
    biases = zeros(h_size)
    i_weights = Variable(Wxh, name="ZMIENNE_WEJSCIE")
    h_weights = Variable(Whh, name="ZMIENNE_UKRYTE")
    o_weights = Variable(Why, name="ZMIENNE_OUT")
    b = Variable(biases, name="BIASES")
    return MyRNNetwork(
        i_weights,
        h_weights,
        o_weights,
        b,
        start_hidden,
        input_size,
        hidden_size,
        output_size
    )
end

function train_net!(net::MyRNNetwork, train_data_x::Any, train_data_y::Any, learning_rate::Float64, batch_size::Int64, epoch_num::Int64)

    println("Training params:")
    println("learning rate -> ", learning_rate)
    println("batch size -> ", batch_size)
    println("number of epochs -> ", epoch_num)

    x1_train = Constant(train_data_x[1:196, 1])
    x2_train = Constant(train_data_x[197:392, 1])
    x3_train = Constant(train_data_x[393:588, 1])
    x4_train = Constant(train_data_x[589:end, 1])
    y_train = Constant(train_data_y[:, 1])

    graph = build_graph(x1_train, x2_train, x3_train, x4_train, net.start_hidden, y_train, net.i_weights, net.h_weights, net.o_weights, net.biases)

    global num_of_clasiffications = 0
    global num_of_correct_clasiffications = 0

    for i in 1:size(train_data_x)[2]
        forward!(graph)
        x1_train.output = train_data_x[1:196, i]
        x2_train.output = train_data_x[197:392, i]
        x3_train.output = train_data_x[393:588, i]
        x4_train.output = train_data_x[589:end, i]
        y_train.output = train_data_y[:, i]
    end

    println("[PRE-TRAIN] acc: ", num_of_correct_clasiffications / num_of_clasiffications, " (", num_of_correct_clasiffications, "/", num_of_clasiffications, ")")

    x1_train = Constant(train_data_x[1:196, 1])
    x2_train = Constant(train_data_x[197:392, 1])
    x3_train = Constant(train_data_x[393:588, 1])
    x4_train = Constant(train_data_x[589:end, 1])
    y_train = Constant(train_data_y[:, 1])
    graph = build_graph(x1_train, x2_train, x3_train, x4_train, net.start_hidden, y_train, net.i_weights, net.h_weights, net.o_weights, net.biases)

    println("Training started...")
    @time for j in 1:epoch_num
        global num_of_clasiffications = 0
        global num_of_correct_clasiffications = 0

        @time for i in 1:size(train_data_x)[2]
            forward!(graph)
            backward!(graph)
            if i % batch_size == 0
                update_weights!(graph, learning_rate, batch_size)
            end
            x1_train.output = train_data_x[1:196, i]
            x2_train.output = train_data_x[197:392, i]
            x3_train.output = train_data_x[393:588, i]
            x4_train.output = train_data_x[589:end, i]
            y_train.output = train_data_y[:, i]
        end
        println("[EPOCH_",j ," TRAIN] acc: ", num_of_correct_clasiffications / num_of_clasiffications, " (", num_of_correct_clasiffications, "/", num_of_clasiffications, ")")
    end

end

function test_net(net::MyRNNetwork, test_data_x::Any, test_data_y::Any)
    x1_test = Constant(test_data_x[1:196, 1])
    x2_test = Constant(test_data_x[197:392, 1])
    x3_test = Constant(test_data_x[393:588, 1])
    x4_test = Constant(test_data_x[589:end, 1])
    y_test = Constant(test_data_y[:, 1])

    test_graph = build_graph(x1_test, x2_test, x3_test, x4_test, net.start_hidden, y_test, net.i_weights, net.h_weights, net.o_weights, net.biases)

    global num_of_correct_clasiffications = 0
    global num_of_clasiffications = 0

    @time for i in 1:size(test_data_x)[2]
        forward!(test_graph)
        x1_test.output = test_data_x[1:196, i]
        x2_test.output = test_data_x[197:392, i]
        x3_test.output = test_data_x[393:588, i]
        x4_test.output = test_data_x[589:end, i]
        y_test.output = test_data_y[:, i]
    end
    println("[TEST] acc: ", num_of_correct_clasiffications / num_of_clasiffications, " (", num_of_correct_clasiffications, "/", num_of_clasiffications, ")")

end
