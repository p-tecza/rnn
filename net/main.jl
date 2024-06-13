include("RNNetwork.jl")

using MLDatasets, Flux

train_data = MLDatasets.MNIST(split=:train)
x1dim = reshape(train_data.features, 28 * 28, :)
yhot = Flux.onehotbatch(train_data.targets, 0:9)

test_data  = MLDatasets.MNIST(split=:test)
x1dim_test = reshape(test_data.features, 28*28, :)
yhot_test = Flux.onehotbatch(test_data.targets, 0:9)

LR = 15e-3
BATCH_SIZE = 100
EPOCHS_NUM = 5

net = init_net(196,64,10);

train_net!(net, x1dim, yhot, LR, BATCH_SIZE, EPOCHS_NUM)
test_net(net, x1dim_test, yhot_test)
