class NeuralNet
  @shape : Array(Int32)
  @outputs : Array(Array(Float64)) | Nil
  @weights : Hash(Int32, Array(Array(Float64))) | Nil
  @weight_update_values : Hash(Int32, Array(Array(Float64))) | Nil
  @weight_changes : Hash(Int32, Array(Array(Float64))) | Nil
  @previous_gradients : Hash(Int32, Array(Array(Float64))) | Nil
  @gradients : Hash(Int32, Array(Array(Float64))) | Nil
  getter :shape, :outputs
  property :weights, :weight_update_values

  DEFAULT_TRAINING_OPTIONS = {
    max_iterations:   1_000,
    error_threshold:  0.01
  }

  def initialize(shape : Array(Int32))
    @shape = shape
  end

  def run(input : Array(Float64)) : Array(Float64)
    # Input to this method represents the output of the first layer (i.e., the input layer)
    @outputs = [input]
    set_initial_weight_values if @weights.nil?

    # Now calculate output of neurons in subsequent layers:
    1.upto(output_layer).each do |layer|
      source_layer = layer - 1 # i.e, the layer that is feeding into this one
      source_outputs = @outputs.not_nil![source_layer]

      if @outputs.not_nil![layer]?.nil?
        @outputs.not_nil! << Array.new(input.size, 0_f64)
      end
      @outputs.not_nil![layer] = @weights.not_nil![layer].map do |neuron_weights|
        # inputs to this neuron are the neuron outputs from the source layer times weights
        inputs = neuron_weights.map_with_index do |weight, i| 
          source_output = source_outputs[i]? || 1 # if no output, this is the bias neuron
          weight * source_output
        end

        sum_of_inputs = inputs.reduce { |sum, n| sum + n }
        # the activated output of this neuron (using sigmoid activation function)
        sigmoid sum_of_inputs
      end
    end

    # Outputs of neurons in the last layer is the final result
    @outputs.not_nil![output_layer]
  end

  def train(inputs, expected_outputs, opts = {} of Symbol => Float64 | Int32)
    opts = DEFAULT_TRAINING_OPTIONS.merge(opts)
    error_threshold, log_every = opts[:error_threshold], opts[:log_every]
    iteration, error = 0, 0

    set_initial_weight_update_values if @weight_update_values.nil?
    set_weight_changes_to_zeros
    set_previous_gradients_to_zeroes

    while iteration < opts[:max_iterations]
      iteration += 1

      error = train_on_batch(inputs, expected_outputs)

      if log_every && (iteration % log_every == 0)
        puts "[#{iteration}] #{(error * 100).round(2)}% mse"
      end

      break if error_threshold && (error < error_threshold)
    end

    {error: error.round(5), iterations: iteration, below_error_threshold: (error < error_threshold)}
  end

  private def train_on_batch(inputs, expected_outputs)
    total_mse = 0

    set_gradients_to_zeroes

    inputs.each_with_index do |input, i|
      run input
      training_error = calculate_training_error expected_outputs[i]
      update_gradients training_error
      total_mse += mean_squared_error training_error
    end

    update_weights

    total_mse / inputs.size.to_f # average mean squared error for batch
  end

  private def calculate_training_error(ideal_output)
    @outputs.not_nil![output_layer].map_with_index do |output, i| 
      output - ideal_output[i]
    end
  end

  private def update_gradients(training_error)
    deltas = {} of Int32 => Array(Float64)
    # Starting from output layer and working backwards, backpropagating the training error
    output_layer.downto(1).each do |layer|
      # deltas[layer] = [] of Float64
      deltas[layer] = Array(Float64).new(@shape[layer], 0_f64)

      @shape[layer].times do |neuron|
        neuron_error = if layer == output_layer
                         -training_error[neuron]
                       else
                         target_layer = layer + 1

                         weighted_target_deltas = deltas[target_layer].map_with_index do |target_delta, target_neuron| 
                           target_weight = @weights.not_nil![target_layer][target_neuron][neuron]
                           target_delta * target_weight
                         end

                         weighted_target_deltas.reduce { |acc, n| acc + n }
                       end

        output = @outputs.not_nil![layer][neuron]
        activation_derivative = output * (1.0 - output)

        delta = deltas[layer][neuron] = neuron_error * activation_derivative

        # gradient for each of this neuron's incoming weights is calculated:
        # the last output from incoming source neuron (from -1 layer)
        # times this neuron's delta (calculated from error coming back from +1 layer)
        source_neurons = @shape[layer - 1] + 1 # account for bias neuron
        source_outputs = @outputs.not_nil![layer - 1]
        gradients = @gradients.not_nil![layer][neuron]

        source_neurons.times do |source_neuron|
          source_output = source_outputs[source_neuron]? || 1 # if no output, this is the bias neuron
          gradient = source_output * delta
          gradients[source_neuron] += gradient # accumulate gradients from batch
        end
      end
    end
  end

  MIN_STEP = Math.exp(-6)
  MAX_STEP = 50_f64

  # Now that we've calculated gradients for the batch, we can use these to update the weights
  # Using the RPROP algorithm - somewhat more complicated than classic backpropagation algorithm, but much faster
  private def update_weights
    1.upto(output_layer) do |layer|
      source_layer = layer - 1
      source_neurons = @shape[source_layer] + 1 # account for bias neuron

      @shape[layer].times do |neuron|
        source_neurons.times do |source_neuron|
          weight_change = @weight_changes.not_nil![layer][neuron][source_neuron]
          weight_update_value = @weight_update_values.not_nil![layer][neuron][source_neuron]
          # for RPROP, we use the negative of the calculated gradient
          gradient = -@gradients.not_nil![layer][neuron][source_neuron]
          previous_gradient = @previous_gradients.not_nil![layer][neuron][source_neuron]

          c = sign(gradient * previous_gradient)

          case c
          when 1 then # no sign change; accelerate gradient descent
            weight_update_value = [weight_update_value * 1.2, MAX_STEP].min
            weight_change = -sign(gradient) * weight_update_value
          when -1 then # sign change; we've jumped over a local minimum
            weight_update_value = [weight_update_value * 0.5, MIN_STEP].max
            weight_change = -weight_change # roll back previous weight change
            gradient = 0_f64 # so won't trigger sign change on next update
          when 0 then
            weight_change = -sign(gradient) * weight_update_value
          end

          @weights.not_nil![layer][neuron][source_neuron] += weight_change
          @weight_changes.not_nil![layer][neuron][source_neuron] = weight_change
          @weight_update_values.not_nil![layer][neuron][source_neuron] = weight_update_value
          @previous_gradients.not_nil![layer][neuron][source_neuron] = gradient
        end
      end
    end
  end

  private def set_weight_changes_to_zeros
    @weight_changes = build_connection_matrixes { 0.0 }
  end

  private def set_gradients_to_zeroes
    @gradients = build_connection_matrixes { 0.0 }
  end

  private def set_previous_gradients_to_zeroes
    @previous_gradients = build_connection_matrixes { 0.0 }
  end

  private def set_initial_weight_update_values
    @weight_update_values = build_connection_matrixes { 0.1 }
  end

  private def set_initial_weight_values
    # Initialize all weights to random float value
    @weights = build_connection_matrixes { rand(-0.5..0.5) }  

    # Update weights for first hidden layer (Nguyen-Widrow method)
    # This is a bit obscure, and not entirely necessary, but it should help the network train faster
    beta = 0.7 * @shape[1]**(1.0 / @shape[0])

    @shape[1].times do |neuron|
      weights = @weights.not_nil![1][neuron]
      norm = Math.sqrt weights.map {|w| w**2}.reduce { |acc, w| acc + w }
      updated_weights = weights.map {|weight| (beta * weight) / norm }
      @weights.not_nil![1][neuron] = updated_weights
    end
  end

  private def build_connection_matrixes
    1.upto(output_layer).reduce({} of Int32 => Array(Array(Float64))) do |hsh, layer|
      # Number of incoming connections to each neuron in this layer:
      source_neurons = @shape[layer - 1] + 1 # == number of neurons in prev layer + a bias neuron

      # matrix[neuron] == Array of values for each incoming connection to neuron
      matrix = Array.new(@shape[layer]) do |neuron|
        Array.new(source_neurons) { yield }
      end

      hsh[layer] = matrix
      hsh
    end
  end

  private def output_layer
    @shape.size - 1
  end

  private def sigmoid(x)
    1 / (1 + Math.exp(-x))
  end

  private def mean_squared_error(errors)
    errors.map {|e| e**2}.reduce { |acc, n| acc + n } / errors.size.to_f
  end

  ZERO_TOLERANCE = Math.exp(-16)

  private def sign(x : Float64) : Float64
    if x > ZERO_TOLERANCE
      1_f64
    elsif x < -ZERO_TOLERANCE
      -1_f64
    else
      0_f64 # x is zero, or a float very close to zero
    end
  end

  private def marshal_dump
    [@shape, @weights, @weight_update_values]
  end

  private def marshal_load(array)
    @shape, @weights, @weight_update_values = array
  end
end
