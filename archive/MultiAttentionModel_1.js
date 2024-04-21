class MultiAttentionModel {
    constructor(inputSize, outputSize, numHeads) {
        this.inputSize = inputSize;
        this.outputSize = outputSize;
        this.numHeads = numHeads;

        // Initialize the attention weights and biases
        this.attentionWeights = [];
        this.attentionBiases = [];
        for (let i = 0; i < numHeads; i++) {
            this.attentionWeights[i] = Matrix.randomize(inputSize, outputSize);
            this.attentionBiases[i] = Matrix.randomize(1, outputSize);
        }

        // Initialize the normalization layer
        this.normalizationLayer = new Matrix(1, outputSize);
    }

    forward(input) {
        // Apply attention mechanism
        this.attentionOutputs = [];
        this.attentionInput = [];

        var attentionOutput;
        for (let i = 0; i < this.numHeads; i++) {
            this.attentionInput.push(input);
            attentionOutput = Matrix.dot(input, this.attentionWeights[i]);
            attentionOutput.add(this.attentionBiases[i]);
            this.attentionOutputs.push(attentionOutput);
            input = attentionOutput;
        }

        // Apply normalization layer
        const normalizedOutput = Matrix.multiplyElementWise(attentionOutput, this.normalizationLayer);

        return normalizedOutput;
    }

    backward(input, target, output, loss, learningRate) {

        // Calculate the gradients for the backward weights and biases
        var gradientsBackwardOutput = Matrix.subtract(target, output);

        // Calculate the gradients for the normalization layer
        const gradientsNormalizationLayer = Matrix.multiply(this.attentionOutputs[this.numHeads - 1], gradientsBackwardOutput);
        this.normalizationLayer.add(gradientsNormalizationLayer.multiplyScaler(learningRate));


        gradientsBackwardOutput = Matrix.multiply(gradientsBackwardOutput, this.normalizationLayer);
        // Calculate the gradients for the attention weights and biases
        for (let i = this.numHeads - 1; i >= 0; i--) {

            // Calculate the gradients for the attention weights
            const gradientsAttentionWeightsLayer = Matrix.dot(this.attentionInput[i].transpose(), gradientsBackwardOutput);

            // Calculate the gradients for the attention biases
            const gradientsAttentionBiasesLayer = gradientsBackwardOutput;

            // Calculate the gradients for the backward output
            gradientsBackwardOutput = Matrix.dot(gradientsBackwardOutput, this.attentionWeights[i].transpose());

            this.attentionWeights[i].add(gradientsAttentionWeightsLayer.multiplyScaler(learningRate));
            this.attentionBiases[i].add(gradientsAttentionBiasesLayer.multiplyScaler(learningRate));
           
        }
    }

    train(input, target, learningRate, numIterations) {
        for (let i = 0; i < numIterations; i++) {
            // Forward pass
            const output = this.forward(input);

            // Calculate the loss
            const loss = this.calculateLoss(output, target);

            // Backward pass
            this.backward(input, target, output, loss, learningRate);

            // Print the loss for monitoring
            console.log(`Iteration ${i + 1}: Loss = ${loss}`);
        }
    }

    calculateLoss(output, target) {
        // Calculate the mean squared error loss
        const error = Matrix.subtract(output, target);
        const squaredError = Matrix.multiplyElementWise(error, error);
        const sumSquaredError = squaredError.data.flat().reduce((acc, val) => acc + val, 0);
        const meanSquaredError = sumSquaredError / (output.rows * output.cols);

        return meanSquaredError;
    }
}
