require 'zlib'
require 'nmatrix'

require_relative './neural_network'
require_relative './neural_network_2'

module Utils

  def self.read_data(dir = './data/')
    image_file = dir + 'train-images-idx3-ubyte.gz'
    label_file = dir + 'train-labels-idx1-ubyte.gz'

    n_rows = n_cols = nil
    images = []
    labels = []

    Zlib::GzipReader.open(image_file) do |f|
      magic, n_images = f.read(8).unpack('N2')
      raise 'This is not MNIST image file' if magic != 2051
      n_rows, n_cols = f.read(8).unpack('N2')
      n_images.times do
        images << f.read(n_rows * n_cols)
      end
    end

    Zlib::GzipReader.open(label_file) do |f|
      magic, n_labels = f.read(8).unpack('N2')
      raise 'This is not MNIST label file' if magic != 2049
      labels = f.read(n_labels).unpack('C*')
    end

    [images, labels]
  end

  def self.prepped_data
    images, labels = read_data

    images.map.with_index do |image, i|
      image_matrix = N[ image.unpack('C*').map { |v| v / 256.0 } ].transpose

      label_matrix = NMatrix.zeros [10, 1]
      label_matrix[labels[i]] = 1.0

      [image_matrix, label_matrix]
    end
  end

  def self.test
    nn = NeuralNetwork.new([784, 30, 10])
    data = prepped_data
    training_set = data.first(50_000)
    test_set = data.last(10_000)

    nn.stochastic_gradient_descent(training_set, 10, 10, 3.0, test_set)

    ex = training_set.sample
    expected = ex[1].to_a.each_with_index.max[1]
    actual = nn.feedforward(ex[0]).to_a.each_with_index.max[1]

    [expected, actual]
  end

  # unable to do matrix mini batch updating
  # http://neuralnetworksanddeeplearning.com/chap2.html#problem_269962
  def self.test2
    nn = NeuralNetwork2.new([784, 30, 10])
    data = prepped_data
    training_set = data.first(50_000)
    test_set = data.last(10_000)

    nn.stochastic_gradient_descent(training_set, 1, 10, 3.0, test_set)

    ex = training_set.sample
    expected = ex[1].to_a.each_with_index.max[1]
    actual = nn.feedforward(ex[0]).to_a.each_with_index.max[1]

    [expected, actual]
  end

end
