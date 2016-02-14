require 'nmatrix'
require 'rubystats'

module BaseFunctions
  def weighted_input(w_l, a_l_n1, b_l)
    w_l.dot(a_l_n1) + b_l
  end

  def sigmoid(z)
    (NMatrix.new(z.shape, Math::E) ** -z + 1) ** -1
  end

  def sigmoid_prime(z)
    sigmoid(z) * (-sigmoid(z) + 1)
  end
end

module QuadraticCost
  extend BaseFunctions

  def self.fn(a, y)
    0.5 * (a - y).norm2 ** 2
  end

  def self.delta(a, y, z)
    (a - y) * sigmoid_prime(z)
  end
end

module CrossEntropyCost
  extend BaseFunctions

  def self.fn(a, y)
    (- y * a.log - (- y + 1) * (- a + 1).log).reduce(&:+)
  end

  def self.delta(a, y, z)
    a - y
  end
end

class NeuralNetwork2
  include BaseFunctions

  def initialize(sizes, cost = CrossEntropyCost)
    @sizes = sizes
    @num_layers = sizes.length
    @cost = cost
    initialize_weights(sizes)
  end

  def initialize_large_weights(sizes)
    @biases = sizes[1..-1].map { |s| N[ rand_array(s) ].transpose }
    @weights = sizes.each_cons(2).map { |n, m| N[ *rand_2d_array(m, n) ] }
  end

  def initialize_weights(sizes)
    @biases = sizes[1..-1].map { |s| N[ rand_array(s) ].transpose }
    @weights = sizes.each_cons(2).map do |n, m|
      std_dev = 1.0 / (n) ** 0.5
      N[ *rand_2d_array(m, n, std_dev) ]
    end
  end

  def feedforward(a)
    (1..(@num_layers - 1)).each do |l|
      a = sigmoid weighted_input(weights(l), a, biases(l))
    end

    a
  end

  def stochastic_gradient_descent(training_set, mini_batch_size, eta, options = {})
    default_options = {
      lmbda: 0.0,
      improvement_threshold: 10
    }
    options = default_options.merge(options)

    epoch = 1
    since_improvement = 0
    last_accuracy = 0.0

    training_size = training_set.length

    while since_improvement <= options[:improvement_threshold] do
      shuffled_t_s = training_set.shuffle

      shuffled_t_s.each_slice(mini_batch_size) do |mini_batch|
        update_mini_batch(mini_batch, eta, options[:lmbda], training_size)
      end

      num_correct = options[:test_set].reduce(0) do |sum, test|
        expected = test[1].to_a.each_with_index.max[1]
        actual = feedforward(test[0]).to_a.each_with_index.max[1]

        sum += 1 if expected == actual
        sum
      end

      accuracy = (num_correct.to_f / options[:test_set].length.to_f) * 100.0

      puts "After epoch #{epoch}: #{accuracy.round(4)}% accuracy"

      if last_accuracy > accuracy
        since_improvement += 1
      else
        since_improvement = 0
      end

      last_accuracy = accuracy

      epoch += 1
    end
  end

  def update_mini_batch(mini_batch, eta, lmbda, training_size)
    nabla_b = @biases.map { |b| NMatrix.zeros b.shape }
    nabla_w = @weights.map { |w| NMatrix.zeros w.shape }

    mini_batch.each do |x, y|
      nabla_b_x, nabla_w_x = backpropagate(x, y)

      nabla_b = nabla_b.map.with_index { |nb, l| nb + nabla_b_x[l] }
      nabla_w = nabla_w.map.with_index { |nw, l| nw + nabla_w_x[l] }
    end

    weight_decay = (1 - eta * lmbda / training_size)
    @biases = @biases.map.with_index { |b, l| b - nabla_b[l] * (eta / mini_batch.length) }
    @weights = @weights.map.with_index { |w, l| w * weight_decay - nabla_w[l] * (eta / mini_batch.length) }
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
    @cost.delta(a_L, y, z_L)
  end

  # TODO: add save and load methods, maybe using pstore

  private

  def weights(l)
    @weights[l - 1]
  end

  def biases(l)
    @biases[l - 1]
  end

  def rand_2d_array(size_x, size_y, std_dev = 1)
    Array.new(size_x) { Array.new(size_y) { norm_dist_random(std_dev) } }
  end

  def rand_array(size)
    Array.new(size) { norm_dist_random(1) }
  end

  def norm_dist_random(std_dev)
    gen = Rubystats::NormalDistribution.new(0, std_dev)
    gen.rng
  end

end
