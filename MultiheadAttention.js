class MultiheadAttention {
    constructor(dModel, numHeads, dropout = 0.1) {
        this.dModel = dModel;
        this.numHeads = numHeads;
        this.headSize = dModel / numHeads;
        this.dropout = dropout;

        // Initialize weights
        this.queryWeights = this._initializeWeights(this.dModel, this.dModel);
        this.keyWeights = this._initializeWeights(this.dModel, this.dModel);
        this.valueWeights = this._initializeWeights(this.dModel, this.dModel);
        this.outputWeights = this._initializeWeights(this.dModel, this.dModel);
    }

    _initializeWeights(rows, cols) {
        const weights = new Matrix(rows, cols);
        weights.randomize(-1 / Math.sqrt(cols), 1 / Math.sqrt(cols)); // Corrected random initialization
        return weights;
    }    

    _initializeZeros(rows, cols) {
        return new Matrix(rows, cols);
    }

    _dropout(input) {
        if (this.dropout === 0) {
            return input;
        }
        const mask = Matrix.randomize(input.rows, input.cols);
        mask.mapInPlace((value) => (value < this.dropout ? 0 : 1));
        const scale = 1 / (1 - this.dropout);
        return (input.multiplyElementWise(mask)).multiplyScaler(scale);
    }

    _dropoutBackward(input, mask) {
        return input.multiplyElementWise(mask).multiplyScaler(1 / (1 - this.dropout));
    }

    forward(query, key, value) {
        const batchSize = query.rows;
    
        this.outputs = [];
        this.inputs = [[query, key, value]];
    
        const queryProjected = this._dropout(query.dot(this.queryWeights));
        const keyProjected = this._dropout(key.dot(this.keyWeights));
        const valueProjected = this._dropout(value.dot(this.valueWeights));
    
        var scaledDotProducts = new Matrix(0, this.dModel);
        for (let i = 0; i < this.numHeads; i++) {
            const [scaledDotProduct] = this.scaledDotProductAttention(
                queryProjected.sliceRow(i * this.headSize, (i + 1) * this.headSize),
                keyProjected.sliceRow(i * this.headSize, (i + 1) * this.headSize),
                valueProjected.sliceRow(i * this.headSize, (i + 1) * this.headSize),
            );
            scaledDotProducts = scaledDotProducts.append(scaledDotProduct);
        }
    
        const output = scaledDotProducts.dot(this.outputWeights);
        this.outputs.push(output);
        return output;
    }    

    visualizeAttentions(query, key, value) {
        const batchSize = query.rows;

        const queryProjected = query.dot(this.queryWeights);
        const keyProjected = key.dot(this.keyWeights);
        const valueProjected = value.dot(this.valueWeights);

        for (let i = 0; i < this.numHeads; i++) {
            this.visualizeAttention(
                queryProjected,
                keyProjected,
                valueProjected,
            );
        }
    }

    backward(gradientsBackwardOutput, learningRate) {
        const gradientsOutputWeights = this.outputs[this.outputs.length - 1].transpose().dot(gradientsBackwardOutput);
        this.outputWeights.add(gradientsOutputWeights.multiplyScaler(learningRate));
        const gradientsBackwardInput = gradientsBackwardOutput.dot(this.outputWeights.transpose());
    
        var gradientsBackwardQueryWeights = new Matrix(0, this.dModel);
        var gradientsBackwardKeyWeights = new Matrix(0, this.dModel);
        var gradientsBackwardValueWeights = new Matrix(0, this.dModel);
    
        for (let i = 0; i < this.numHeads; i++) {
            const start = i * this.headSize;
            const end = (i + 1) * this.headSize;
            const [gradientsQuery, gradientsKey, gradientsValue] = this.scaledDotProductAttentionBackward(
                gradientsBackwardInput.sliceRow(start, end),
                this.inputs[0][0].sliceRow(start, end),
                this.inputs[0][1].sliceRow(start, end),
                this.inputs[0][2].sliceRow(start, end),
            );
    
            gradientsBackwardQueryWeights = gradientsBackwardQueryWeights.append(gradientsQuery);
            gradientsBackwardKeyWeights = gradientsBackwardKeyWeights.append(gradientsKey);
            gradientsBackwardValueWeights = gradientsBackwardValueWeights.append(gradientsValue);
        }
    
        const learningRateScaled = learningRate / this.numHeads;
    
        this.queryWeights.add(gradientsBackwardQueryWeights.multiplyScaler(learningRateScaled));
        this.keyWeights.add(gradientsBackwardKeyWeights.multiplyScaler(learningRateScaled));
        this.valueWeights.add(gradientsBackwardValueWeights.multiplyScaler(learningRateScaled));
    }    
    
    

    visualizeAttention(query, key, value) {
        const sqrtHeadSize = Math.sqrt(this.headSize);
    
        // Calculate attention scores
        const scores = this.calculateAttentionScores(query, key);
    
        // Softmax to get attention weights
        const attentionWeights = scores.softmax();
    
        // Dot product of attention weights with value
        const context = attentionWeights.dot(value);
    
        console.log("Attention scores: ", scores.toArray(), context);
    
        // Prepare data for plotting
        const scoreArray = scores.toArray();
        const weightArray = attentionWeights.toArray();
    
        // Create canvas
        const canvas = document.createElement("canvas");
        canvas.width = 600;
        canvas.height = 800;
        document.body.appendChild(canvas);
        const ctx = canvas.getContext("2d");
    
        // Clear canvas
        ctx.clearRect(0, 0, canvas.width, canvas.height);
    
        // Calculate maximum attention score for normalization
        const maxScore = Math.max(...scoreArray);
        const maxWeight = Math.max(...weightArray);
    
        // Plot attention scores
        for (let i = 0; i < scoreArray.length; i++) {
            const x = 50 + i * 20;
            const height = (scoreArray[i] / maxScore) * 150;
            if (!isFinite(height)) continue; // Skip if the height is not finite
            const gradientScore = ctx.createLinearGradient(x, 50, x, 50 - height);
            gradientScore.addColorStop(0, 'blue');
            gradientScore.addColorStop(1, 'lightblue');
            ctx.fillStyle = gradientScore;
            ctx.fillRect(x, 50, 10, -height);
        }
    
        // Plot attention weights
        for (let i = 0; i < weightArray.length; i++) {
            const x = 50 + i * 20;
            const height = (weightArray[i] / maxWeight) * 150;
            if (!isFinite(height)) continue; // Skip if the height is not finite
            const gradientWeight = ctx.createLinearGradient(x, 200, x, 200 - height);
            gradientWeight.addColorStop(0, 'red');
            gradientWeight.addColorStop(1, 'pink');
            ctx.fillStyle = gradientWeight;
            ctx.fillRect(x, 200, 10, -height);
        }
    
        return context;
    }
    

    calculateAttentionScores(query, key) {
        const sqrtHeadSize = Math.sqrt(this.headSize);
        return query.dot(key.transpose()).multiplyScaler(1 / sqrtHeadSize);
    }

    scaledDotProductAttention(query, key, value) {
        const scores = this.calculateAttentionScores(query, key);
        const attentionWeights = scores.softmax();
        return [attentionWeights.dot(value), attentionWeights];
    }    

    scaledDotProductAttentionBackward(gradientsBackwardOutput, query, key, value) {            
        // Compute gradients with respect to attention weights
        const gradientsAttentionWeights = gradientsBackwardOutput.dot(value.transpose());
    
        const scores = this.calculateAttentionScores(query, key);
        const attentionWeights = scores.softmax();
    
        // Compute gradients with respect to scores
        const gradientsScores = gradientsAttentionWeights.softmaxBackward(attentionWeights).multiplyScaler(1 / this.headSize);
    
        // Compute gradients with respect to query, key, and value
        const gradientsQuery = gradientsScores.dot(key);
        const gradientsKey = gradientsScores.dot(query);
        const gradientsValue = attentionWeights.transpose().dot(gradientsBackwardOutput);
    
        return [gradientsQuery, gradientsKey, gradientsValue];
    }
    
    
    save() {
        return {
            queryWeights: this.queryWeights,
            keyWeights: this.keyWeights,
            valueWeights: this.valueWeights,
            outputWeights: this.outputWeights,
        };
    }
}
