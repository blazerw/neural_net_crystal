require "./spec_helper"
require "gzip"

Spec2.describe NeuralNetCrystal do
  context "examples" do
    # ########################################################################
    # # Iris Species Predictor
    # ########################################################################
    it "predicts iris species" do
      # This neural network will predict the species of an iris based on sepal and petal size
      # Dataset: http://en.wikipedia.org/wiki/Iris_flower_data_set

      rows = File.read_lines("./spec/resources/iris.csv").map {|l| l.chomp.split(',') }

      rows.shuffle!

      label_encodings = {
        "Iris-setosa"     => [1, 0, 0],
        "Iris-versicolor" => [0, 1, 0],
        "Iris-virginica"  => [0, 0 ,1]
      }

      x_data = rows.map {|row| row[0,4].map(&.to_f64) }
      y_data = rows.map {|row| label_encodings[row[4]] }

      # Normalize data values before feeding into network
      # maps input to float between 0 and 1
      normalize = -> (val : Float64, high : Float64, low : Float64) {  (val - low) / (high - low) }

      columns = (0..3).map do |i|
        x_data.map {|row| row[i] }
      end

      x_data.map! do |row|
        row.map_with_index do |val, j|
          max, min = columns[j].max, columns[j].min
          normalize.call(val, max, min)
        end
      end

      x_train = x_data[0, 100]
      y_train = y_data[0, 100]

      x_test = x_data[100, 50]
      y_test = y_data[100, 50]

      # Build a 3 layer network: 4 input neurons, 4 hidden neurons, 3 output neurons
      # Bias neurons are automatically added to input + hidden layers; no need to specify these
      nn = NeuralNet.new [4_i32, 4_i32, 3_i32]

      prediction_success = -> (actual : Array(Float64), ideal : Array(Int32)) {
        predicted = (0..2).max_by {|i| actual[i] }
        ideal[predicted] == 1
      }

      mse = -> (actual : Array(Float64), ideal : Array(Int32)) {
        errors = actual.zip(ideal).map {|a, i| a - i }
        (errors.reduce(0) {|sum, err| sum += err**2}) / errors.size.to_f64
      }

      error_rate = -> (errors : Int32, total : Int32) { ((errors.to_f64 / total.to_f64) * 100_f64).round }

      run_test = -> (nn : NeuralNet, inputs : Array(Array(Float64)), expected_outputs : Array(Array(Int32))) {
        success, failure, errsum = 0_i32, 0_i32, 0_i32
        inputs.each_with_index do |input, i|
          output = nn.run(input)
          prediction_success.call(output, expected_outputs[i]) ? (success = success + 1) : (failure = failure + 1)
          errsum = errsum + mse.call(output, expected_outputs[i])
        end
        [success, failure, errsum / inputs.size.to_f64]
      }

      puts "Testing the untrained network..."

      success, failure, avg_mse = run_test.call(nn, x_test, y_test)

      puts "Untrained classification success: #{success}, failure: #{failure} (classification error: #{error_rate.call(failure.to_i32, x_test.size)}%, mse: #{(avg_mse * 100).round(2)}%)"


      puts "\nTraining the network...\n\n"

      t1 = Time.now
      result = nn.train(x_train, y_train, { error_threshold: 0.01,
                        max_iterations: 1_000,
                        log_every: 100 }
                       )

      puts "\nDone training the network: #{result[:iterations]} iterations, #{(result[:error] * 100).round(2)}% mse, #{(Time.now - t1).total_seconds.round(1)}s"


      puts "\nTesting the trained network..."

      success, failure, avg_mse = run_test.call(nn, x_test, y_test)

      puts "Trained classification success: #{success}, failure: #{failure} (classification error: #{error_rate.call(failure.to_i32, x_test.size)}%, mse: #{(avg_mse * 100).round(2)}%)"

      expect(success > failure).to be_truthy
    end

    # ########################################################################
    # # MNIST (OCR)                                                          #
    # ########################################################################
    it "OCRs hadwriting" do
      # This neural net performs OCR on handwritten digits from the MNIST dataset
      # MNIST datafiles can be downloaded here: http://yann.lecun.com/exdb/mnist/

      mnist_images_file = "spec/resources/train-images-idx3-ubyte.gz"
      mnist_labels_file = "spec/resources/train-labels-idx1-ubyte.gz"

      unless File.exists?(mnist_images_file) && File.exists?(mnist_labels_file)
        raise "Missing MNIST datafiles\nMNIST datafiles must be present in an mnist/ directory\nDownload from: http://yann.lecun.com/exdb/mnist/"
      end

      # MNIST loading code adapted from here:
      # https://github.com/shuyo/iir/blob/master/neural/mnist.rb
      # (Well, the Ruby code used the above, the Crystal came next.)
      n_rows = n_cols : UInt32 | Nil = nil
      images = [] of Bytes
      labels = [] of Int32
      File.open(mnist_images_file) do |f|
        Gzip::Reader.open(f) do |z|
          magic = z.read_bytes(UInt32, IO::ByteFormat::BigEndian)
          raise "This is not MNIST image file" if magic != 2051
          n_images = z.read_bytes(UInt32, IO::ByteFormat::BigEndian)
          n_rows = z.read_bytes(UInt32, IO::ByteFormat::BigEndian)
          n_cols = z.read_bytes(UInt32, IO::ByteFormat::BigEndian)
          n_images.times do
            image_bytes = Bytes.new(n_rows * n_cols)
            z.read_fully(image_bytes)
            images << image_bytes
          end
        end
      end

      File.open(mnist_labels_file) do |f|
        Gzip::Reader.open(f) do |z|
          magic = z.read_bytes(UInt32, IO::ByteFormat::BigEndian)
          raise "This is not MNIST label file" if magic != 2049
          n_labels = z.read_bytes(UInt32, IO::ByteFormat::BigEndian)
          n_labels.times do
            labels << (z.read_byte || 0_u8).to_i32
          end
        end
      end

      # collate image and label data
      data = images.map_with_index do |image, i|
        target = [0]*10
        target[labels[i]] = 1
        {image, target}
      end

      # data.shuffle!

      train_size = (ARGV[0]? || 100).to_i
      test_size = 100
      hidden_layer_size = (ARGV[1]? || 25).to_i

      # maps input to float between 0 and 1
      normalize = -> (val : Int32, fromLow : Int32, fromHigh : Int32, toLow : Int32, toHigh : Int32) {  (val - fromLow) * (toHigh - toLow) / (fromHigh - fromLow).to_f }

      x_data = Array(Array(Float64)).new(train_size + test_size)
      y_data = Array(Array(Int32)).new(train_size + test_size)

      data[0, train_size + test_size].each do |row|
        image = row[0].map {|v| normalize.call(v.to_i32, 0, 256, 0, 1)}.to_a
        x_data << image
        y_data << row[1]
      end

      x_train = x_data[0, train_size]
      y_train = y_data[0, train_size]

      x_test = x_data[train_size, test_size]
      y_test = y_data[train_size, test_size]


      puts "Initializing network with #{hidden_layer_size} hidden neurons."
      nn = NeuralNet.new [28*28, hidden_layer_size, 50, 10]

      error_rate = -> (errors : Int32, total : Int32) { ((errors.to_f64 / total.to_f64) * 100_f64).round }

      mse = -> (actual : Array(Float64), ideal : Array(Int32)) {
        errors = actual.zip(ideal).map {|a, i| a - i }
        (errors.reduce(0) {|sum, err| sum += err**2}) / errors.size.to_f
      }

      decode_output = -> (output : Array(Float64) | Array(Int32)) { (0..9).max_by {|i| output[i]} }
      prediction_success = -> (actual : Array(Float64), ideal : Array(Int32)) {
        decode_output.call(actual) == decode_output.call(ideal)
      }

      run_test = -> (nn : NeuralNet, inputs : Array(Array(Float64)), expected_outputs : Array(Array(Int32))) {
        success, failure, errsum = 0,0,0
        inputs.each.with_index do |input, i|
          output = nn.run input
          prediction_success.call(output, expected_outputs[i]) ? (success = success + 1) : (failure = failure + 1)
          errsum += mse.call(output, expected_outputs[i])
        end
        [success, failure, errsum / inputs.size.to_f]
      }

      puts "Testing the untrained network..."

      success, failure, avg_mse = run_test.call(nn, x_test, y_test)

      puts "Untrained classification success: #{success}, failure: #{failure} (classification error: #{error_rate.call(failure.to_i32, x_test.size)}%, mse: #{(avg_mse * 100).round(2)}%)"

      puts "\nTraining the network with #{train_size} data samples...\n\n"
      t = Time.now
      result = nn.train(x_train, y_train, { log_every: 1, max_iterations: 100, error_threshold:  0.01 })

      puts "\nDone training the network: #{result[:iterations]} iterations, #{(result[:error] * 100).round(2)}% mse, #{(Time.now - t).total_seconds.round(1)}s"

      # # Marshal test
      # dumpfile = 'mnist/network.dump'
      # File.write(dumpfile, Marshal.dump(nn))
      # nn = Marshal.load(File.read(dumpfile))

      puts "\nTesting the trained network..."

      success, failure, avg_mse = run_test.call(nn, x_test, y_test)

      # puts result
      puts "Trained classification success: #{success}, failure: #{failure} (classification error: #{error_rate.call(failure.to_i32, x_test.size)}%, mse: #{(avg_mse * 100).round(2)}%)"

      expect(success > failure).to be_truthy
#
#      # require_relative './image_grid'
#      # ImageGrid.new(nn.weights[1]).to_file 'examples/mnist/hidden_weights.png'
    end

    # ########################################################################
    # # Miles per Gallon                                                     #
    # ########################################################################
    it "miles per gallons" do
      # Dataset available at https://archive.ics.uci.edu/ml/datasets/Auto+MPG
      rows = [] of Array(String)
      File.each_line("spec/resources/auto-mpg.data") do |line|
        csv_line = line.gsub(/  +/, ",")
        new_line = csv_line.split(",")
        rows << new_line
      end

      rows.shuffle!

      # maps input to float between 0 and 1
      normalize = -> (val : Float64, fromLow : Int32, fromHigh : Int32, toLow : Int32, toHigh : Int32) {  (val - fromLow) * (toHigh - toLow) / (fromHigh - fromLow).to_f }

      x_data = [] of Array(Float64)
      y_data = [] of Array(Float64)

      rows.each do |row|
        mpg = normalize.call(row[0].to_f, 0, 60, 0, 1)
        cylinders = normalize.call(row[1].to_f, 2, 12, 0, 1)
        displacement = normalize.call(row[2].to_f, 10, 1000, 0, 1)
        horsepower = normalize.call(row[3].to_f, 1, 1000, 0, 1)
        weight = normalize.call(row[4].to_f, 100, 5000, 0, 1)

        x_data << [cylinders, displacement, horsepower, weight]
        y_data << [mpg]
      end

      test_size = 100
      train_size = rows.size - test_size

      x_train = x_data[0, train_size]
      y_train = y_data[0, train_size]
      x_test = x_data[train_size, test_size]
      y_test = y_data[train_size, test_size]

      to_mpg = -> (flt : Float64) { normalize.call(flt, 0, 1, 0, 60) }

      mse = -> (actual : Array(Float64), ideal : Array(Float64)) {
        errors = actual.zip(ideal).map {|a, i| a - i }
        (errors.reduce(0) {|sum, err| sum += err**2}) / errors.size.to_f
      }

      run_test = -> (nn : NeuralNet, inputs : Array(Array(Float64)), expected_outputs : Array(Array(Float64))) {
        mpg_err, errsum = 0, 0
        outputs = [] of Array(Float64)

        inputs.each.with_index do |input, i|
          output = nn.run input
          outputs << output
          mpg_err = mpg_err + (to_mpg.call(output[0]) - to_mpg.call(expected_outputs[i][0])).abs

          errsum += mse.call(output, expected_outputs[i])
        end

        y_mean = expected_outputs.reduce(0.0) { |sum, val| sum + val[0] } / expected_outputs.size
        y_sum_squares = expected_outputs.map{|val| (val[0] - y_mean)**2 }.reduce { |sum, n| sum + n }
        y_residual_sum_squares = outputs.zip(expected_outputs).map {|out, expected| (expected[0] - out[0])**2 }.reduce { |sum, n| sum + n }
        r_squared = 1.0 - (y_residual_sum_squares / y_sum_squares)

        [mpg_err / inputs.size.to_f, errsum / inputs.size.to_f, r_squared]
      }

      show_examples = -> (nn : NeuralNet, x : Array(Array(Float64)), y : Array(Array(Float64))) {
        puts "Actual\tPredict\tError (mpg)"
        10.times do |i|
          output = nn.run x[i]
          predicted = to_mpg.call(output[0])
          actual = to_mpg.call(y[i][0])
          puts "#{actual.round(1)}\t#{predicted.round(1)}\t#{(predicted - actual).abs.round(1)}"
        end
      }

      nn = NeuralNet.new [4, 4, 1]

      puts "Testing the untrained network..."
      mpg_err_b, avg_mse, r_squared = run_test.call(nn, x_test, y_test)
      puts "Average prediction error: #{mpg_err_b.round(2)} mpg (mse: #{(avg_mse * 100).round(2)}%, r-squared: #{r_squared.round(2)})"

      # puts "\nUntrained test examples (first 10):"
      # show_examples.(nn, x_test, y_test)

      puts "\nTraining the network...\n\n"
      t1 = Time.now
      result = nn.train(x_train, y_train, { error_threshold: 0.005,
                        max_iterations: 100,
                        log_every: 10 }
                       )

      # puts result
      puts "\nDone training the network: #{result[:iterations]} iterations, #{(result[:error] * 100).round(2)}% mse, #{(Time.now - t1).total_seconds.round(1)}s"

      puts "\nTesting the trained network..."
      mpg_err_a, avg_mse, r_squared = run_test.call(nn, x_test, y_test)
      puts "Average prediction error: #{mpg_err_a.round(2)} mpg (mse: #{(avg_mse * 100).round(2)}%, r-squared: #{r_squared.round(2)})"

      puts "\nTrained test examples (first 10):"
      show_examples.call(nn, x_test, y_test)

      expect(mpg_err_a < mpg_err_b).to be_truthy
    end
  end
end
