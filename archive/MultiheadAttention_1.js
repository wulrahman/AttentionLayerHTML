
class MultiheadAttention {
    constructor(dModel, numHeads) {
        this.dModel = dModel;
        this.numHeads = numHeads;
        this.headSize = dModel / numHeads;

        this.queryWeights = Matrix.randomize(dModel, dModel)
        this.keyWeights = Matrix.randomize(dModel, dModel);
        this.valueWeights = Matrix.randomize(dModel, dModel);
        this.outputWeights = Matrix.randomize(dModel, dModel);
    }

    forward(query, key, value) {
        const batchSize = query.rows;

        const queryProjected = query.dot(this.queryWeights);
        const keyProjected = key.dot(this.keyWeights);
        const valueProjected = value.dot(this.valueWeights);

        const querySplit = queryProjected.split(1, 0, this.numHeads - 1);
        const keySplit = keyProjected.split(1, 0, this.numHeads - 1);
        const valueSplit = valueProjected.split(1, 0, this.numHeads - 1);

        var scaledDotProducts = new Matrix(0, this.dModel);
        for (let i = 0; i < this.numHeads; i++) {
            const scaledDotProduct = this.scaledDotProductAttention(
                querySplit.getRow(i),
                keySplit.getRow(i),
                valueSplit.getRow(i),
            );
            scaledDotProducts = scaledDotProducts.append(scaledDotProduct);
        }

        // const concatenated = Matrix.concatenate(scaledDotProducts);
        const output = scaledDotProducts.dot(this.outputWeights);

        return output;
    }

    backward(input, target, output, loss, learningRate) {
        const gradientsBackwardOutput = Matrix.subtract(target, output);
        gradientsBackwardOutput.normalize();

        const gradientsBackwardInput = gradientsBackwardOutput.dot(this.outputWeights.transpose());

        const inputTranspose = input.transpose();
        const gradientsOutputWeights = inputTranspose.dot(gradientsBackwardOutput);
        this.outputWeights.add(gradientsOutputWeights.multiplyScaler(learningRate));



        const gradientsBackwardInputWeights = [];
        for (let i = 0; i < this.numHeads; i++) {
            var query = (this.queryWeights.getRow(i)).transpose();
            query = query.dot(gradientsBackwardInput.getRow(i));
            var key = (this.keyWeights.getRow(i)).transpose();
            key = key.dot(gradientsBackwardInput.getRow(i));
            var value = (this.valueWeights.getRow(i)).transpose();
            value = value.dot(gradientsBackwardInput.getRow(i));

            query.add(key);
            query.add(value);
            
            gradientsBackwardInputWeights.push(query);
        }

        for (let i = 0; i < this.numHeads; i++) {
            this.queryWeights.add(gradientsBackwardInputWeights[i].multiplyScaler(learningRate));
            this.keyWeights.add(gradientsBackwardInputWeights[i].multiplyScaler(learningRate));
            this.valueWeights.add(gradientsBackwardInputWeights[i].multiplyScaler(learningRate));
        }
    }

    scaledDotProductAttention(query, key, value) {
        const sqrtHeadSize = Math.sqrt(this.headSize);

        const scores = query.dot(key.transpose()).multiplyScaler(1 / sqrtHeadSize);
        const attentionWeights = scores.softmax();

        const context = attentionWeights.dot(value);

        return context;
    }
}

const dModel = 20;
const numHeads = 20;

const attention = new MultiheadAttention(dModel, numHeads);

const query = Matrix.randomize(dModel, numHeads);
const key = Matrix.randomize(dModel, numHeads)
const value = Matrix.randomize(dModel, numHeads);

const target = Matrix.randomize(dModel, numHeads);

var loss;
for (let i = 0; i < 1000; i++) {
    const output = attention.forward(query, key, value);

    loss = output.meanSquaredError(target);
    attention.backward(query, target, output, loss, 0.00001);
    console.log(loss);

}

console.log(target);
const output = attention.forward(query, key, value);
console.log(output);

