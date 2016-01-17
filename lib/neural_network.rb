require 'nmatrix'
require 'rubystats'

class NeuralNetwork

  def initialize(sizes)
    @sizes = sizes
    @num_layers = sizes.length
    @biases = sizes[1..-1].map { |s| N[ rand_array(s) ].transpose }
    @weights = sizes.each_cons(2).map { |n, m| N[ *rand_2d_array(m, n) ] }
  end

  def feedforward(a)
    (1..(@num_layers - 1)).each do |l|
      a = sigmoid weighted_input(weights(l), a, biases(l))
    end

    a
  end

  def stochastic_gradient_descent(training_set, epochs, mini_batch_size, eta, test_set = nil)
    (1..epochs).each do |epoch|
      shuffled_t_s = training_set.shuffle

      shuffled_t_s.each_slice(mini_batch_size) do |mini_batch|
        update_mini_batch(mini_batch, eta)
      end

      if test_set
        num_correct = test_set.reduce(0) do |sum, test|
          expected = test[1].to_a.each_with_index.max[1]
          actual = feedforward(test[0]).to_a.each_with_index.max[1]

          sum += 1 if expected == actual
          sum
        end

        puts "After #{epoch} epoch: #{num_correct} / #{test_set.length}"
      end
    end
  end

  def update_mini_batch(mini_batch, eta)
    nabla_b = @biases.map { |b| NMatrix.zeros b.shape }
    nabla_w = @weights.map { |w| NMatrix.zeros w.shape }

    mini_batch.each do |x, y|
      nabla_b_x, nabla_w_x = backpropagate(x, y)

      nabla_b = nabla_b.map.with_index { |nb, l| nb + nabla_b_x[l] }
      nabla_w = nabla_w.map.with_index { |nw, l| nw + nabla_w_x[l] }
    end

    @biases = @biases.map.with_index { |b, l| b - nabla_b[l] * (eta / mini_batch.length) }
    @weights = @weights.map.with_index { |w, l| w - nabla_w[l] * (eta / mini_batch.length) }
  end

  def backpropagate(x, y)
    # feedforward
    output_activation = x
    activations = [x]
    # first weighted input in not used
    # nil is necessary to keep indexes correct
    weighted_inputs = [nil]

    (1..(@num_layers - 1)).each do |l|
      weighted_input_l = weighted_input weights(l), output_activation, biases(l)
      weighted_inputs << weighted_input_l

      output_activation = sigmoid weighted_input_l
      activations << output_activation
    end

    # output error
    delta = output_error(output_activation, y, weighted_inputs.last)

    # backpropagate the error
    nabla_b = Array.new @biases.length
    nabla_b[-1] = delta

    nabla_w = Array.new @weights.length
    nabla_w[-1] = delta.dot activations[-2].transpose

    (1..(@num_layers - 2)).to_a.reverse.each do |l|
      delta = backprop_error(weights(l + 1), delta, weighted_inputs[l])

      nabla_b[l - 1] = delta
      nabla_w[l - 1] = delta.dot activations[l - 1].transpose
    end

    [nabla_b, nabla_w]
  end

  def backprop_error(w_l_p1, delta_l_p1, z_l)
    w_l_p1.transpose.dot(delta_l_p1) * sigmoid_prime(z_l)
  end

  def output_error(a_L, y, z_L)
    (a_L - y) * sigmoid_prime(z_L)
  end

  def weighted_input(w_l, a_l_n1, b_l)
    w_l.dot(a_l_n1) + b_l
  end

  def sigmoid(z)
    (NMatrix.new(z.shape, Math::E) ** -z + 1) ** -1
  end

  def sigmoid_prime(z)
    sigmoid(z) * (-sigmoid(z) + 1)
  end

  private

  def weights(l)
    @weights[l - 1]
  end

  def biases(l)
    @biases[l - 1]
  end

  def rand_2d_array(size_x, size_y)
    Array.new(size_x) { Array.new(size_y) {norm_dist_random} }
  end

  def rand_array(size)
    Array.new(size) { norm_dist_random }
  end

  def norm_dist_random
    gen = Rubystats::NormalDistribution.new(0, 1)
    gen.rng
  end

end
