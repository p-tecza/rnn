# function sigmoid(x::Float64)
#     return exp(x) / (exp(x) + 1.0)
# end

# function tanh(x::Float64)
#     return (exp(x) - exp(-x))/(exp(x) + exp(-x))
# end

function pass!(input::Float64, short_term_mem::Float64, long_term_mem::Float64, input_weights::Vector{Float64}, mem_weights::Vector{Float64})

# IGNORING BIASES

# forget gate

forget_sum::Float64 = short_term_mem * mem_weights[1] + input * input_weights[1]
remember_percent::Float64 = sigmoid(forget_sum)
long_term_mem = long_term_mem * remember_percent

# input gate

potential_memory_sum = short_term_mem * mem_weights[2] + input * input_weights[2]
pot_mem_sum_sig = sigmoid(potential_memory_sum)
potential_LT_memory_sum = short_term_mem * mem_weights[3] + input * input_weights[3]
pot_LT_mem_sum_tanh = tanh(potential_LT_memory_sum)
input_gate_product = pot_LT_mem_sum_tanh * pot_mem_sum_sig
long_term_mem = long_term_mem + input_gate_product

# output gate

potential_ST_memory_sum = short_term_mem * mem_weights[4] + input * input_weights[4]
pot_ST_mem_sum_sig = sigmoid(potential_ST_memory_sum)
short_term_mem = tanh(long_term_mem) * pot_ST_mem_sum_sig
return (short_term_mem, long_term_mem)
end

@show sigmoid(1.)

@show tanh(0.88)



res = pass!(1.,0.,0.,[.5,.3,.2,.1], [.7,.5,.3,.1])
@show res
res2 = pass!(.3,res[1],res[2],[.5,.3,.2,.1], [.7,.5,.3,.1])
@show res2
