class BuildHuffmanGramModel {
    constructor(embeddingSize) {
        this.wordVectors = []; // Associative array to store word vectors
        this.vocabSize = 0;
        this.embeddingSize = embeddingSize;

        this.wordToIndex = {};

        this.negativeSamples = 0; // Define the number of negative samples

        this.wordFrequencies = {};

        this.huffmanParent = {};
        this.subsamplingThreshold = 1e-3;

        this.wordToIndex[' '] = 0;
        this.wordVectors[0] = this.zeros(1, this.embeddingSize)[0];
        this.wordFrequencies[0] = 1;
    }

    zeros(rows, cols) {
        const matrix = [];
        for (let i = 0; i < rows; i++) {
            const row = [];
            for (let j = 0; j < cols; j++) {
                row.push(0);
            }
            matrix.push(row);
        }
        return matrix;
    }

    getBinaryPath(contextWordIndex) {
        const path = [];

        while (contextWordIndex !== null) {
            path.push(contextWordIndex);
            contextWordIndex = this.huffmanParent[contextWordIndex] ?? null; // Navigate to parent node
        }

        return path;
    }

    forwardPropagation(centerWordIndex, contextWordIndex) {
        const path = this.getBinaryPath(contextWordIndex); // Get the binary path to the context word

        let prob = 1.0;
        for (const nodeIndex of path) {
            console.log(this.wordVectors[centerWordIndex], this.wordVectors[nodeIndex])

            prob *= this.sigmoid(this.dotProduct(this.wordVectors[centerWordIndex], this.wordVectors[nodeIndex]));
        }

        return prob;
    }

    findClosestWord(sumVector) {
        let closestWord = null;
        let maxSimilarity = -1; // Initialize to a low value

        for (const [word, wordVector] of Object.entries(this.wordVectors)) {
            const similarity = this.cosineSimilarity(sumVector, wordVector);

            if (similarity > maxSimilarity) {
                maxSimilarity = similarity;
                closestWord = word;
            }
        }

        return this.wordToIndex[closestWord];
    }

    dotProduct(vector1, vector2) {
        let result = 0;
        const length = vector1.length; // Assuming both vectors have the same length

        for (let i = 0; i < length; i++) {
            result += vector1[i] * vector2[i];
        }

        return result;
    }

    buildHuffmanTree() {
        this.vocabSize = Object.keys(this.wordVectors).length;

        const wordFrequencies = this.wordFrequencies; // Replace this with your actual word frequencies

        // Create initial nodes for each word
        let nodes = [];
        for (let i = 0; i < this.vocabSize; i++) {
            nodes.push({ index: i, frequency: wordFrequencies[i] ?? 0 });
        }

        // Build the Huffman tree
        while (nodes.length > 1) {
            nodes.sort((a, b) => a.frequency - b.frequency);

            const min1 = nodes.shift();
            const min2 = nodes.shift();

            // Create a new node representing the merged nodes
            const mergedNode = { index: null, frequency: min1.frequency + min2.frequency, children: [min1, min2] };

            // Add the merged node back to the list of nodes
            nodes.push(mergedNode);
        }

        // Populate huffmanParent based on the tree structure
        this.populateHuffmanParent(nodes[0], null);
    }

    populateHuffmanParent(node, parentIndex) {
        if (node.index !== null) {
            // This is a leaf node
            this.huffmanParent[node.index] = parentIndex;
        } else {
            // This is an internal node with children
            for (const childNode of node.children) {
                this.populateHuffmanParent(childNode, node.index);
            }
        }
    }

    sigmoid(x) {
        return 1 / (1 + Math.exp(-x));
    }

    backwardPropagation(centerWordIndex, contextWordIndex, learningRate) {
        this.vocabSize = Object.keys(this.wordVectors).length;

        const predictedProbability = this.forwardPropagation(centerWordIndex, contextWordIndex);
        const gradient = predictedProbability - 1;

        const path = this.getBinaryPath(contextWordIndex);

        for (const nodeIndex of path) {
            for (let i = 0; i < this.embeddingSize; i++) {
                this.wordVectors[centerWordIndex][i] -= learningRate * gradient * this.wordVectors[nodeIndex][i];
                this.wordVectors[nodeIndex][i] -= learningRate * gradient * this.wordVectors[centerWordIndex][i];
            }
        }
    }

    train(trainingPairs, epochs, initialLearningRate) {
        let learningRate = initialLearningRate;

        for (let epoch = 0; epoch < epochs; epoch++) {
            for (const pair of trainingPairs) {
                const centerWordIndex = pair[0];
                const contextWordIndex = pair[1];

                // Adjust learning rate dynamically
                learningRate *= 0.95; // Reduce the learning rate by a factor

                // Apply subsampling
                if (Math.random() < this.getSubsamplingProbability(contextWordIndex)) {
                    continue; // Skip this context word
                }

                // Use the adjusted learning rate in backwardPropagation
                this.backwardPropagation(centerWordIndex, contextWordIndex, learningRate);
            }
        }
    }

    evaluateSimilarity(word1, word2) {
        const vector1 = this.wordVectors[word1];
        const vector2 = this.wordVectors[word2];
        return this.cosineSimilarity(vector1, vector2);
    }

    initializeRandomVector() {
        const vector = [];
        for (let i = 0; i < this.embeddingSize; i++) {
            vector.push((Math.random() - 0.5) / this.embeddingSize);
        }
        return vector;
    }

    cosineSimilarity(vectorA, vectorB) {
        let dotProduct = 0;
        let normA = 0;
        let normB = 0;

        for (let i = 0; i < this.embeddingSize; i++) {
            dotProduct += vectorA[i] * vectorB[i];
            normA += vectorA[i] * vectorA[i];
            normB += vectorB[i] * vectorB[i];
        }

        if (normA === 0 || normB === 0) {
            return 0; // Handle division by zero
        }

        return dotProduct / (Math.sqrt(normA) * Math.sqrt(normB));
    }

    generateTrainingPairs(vocabulary, contextWindowSize) {
        const trainingPairs = [];

        for (const [key, value] of Object.entries(vocabulary)) {
            let word = value.toLowerCase();
            word = word.replace(/[^a-z0-9]+/gi, '');

            if (!(word in this.wordToIndex)) {
                // Initialize word vectors randomly
                this.wordVectors[this.vocabSize] = this.initializeRandomVector();
                this.wordToIndex[word] = this.vocabSize;
                this.wordFrequencies[this.wordToIndex[word]] = 1;
                this.vocabSize++;
            } else {
                this.wordFrequencies[this.wordToIndex[word]]++;
            }
        }

        for (let i = contextWindowSize; i < vocabulary.length - contextWindowSize; i++) {
            const centerWordIndex = i;

            for (let j = i - contextWindowSize; j <= i + contextWindowSize; j++) {
                if (j !== i) {
                    const contextWordIndex = j;
                    trainingPairs.push([centerWordIndex, contextWordIndex]);
                }
            }
        }

        this.buildHuffmanTree();

        return trainingPairs;
    }

    displayTrainingPairs(vocabulary, trainingPairs) {
        for (const pair of trainingPairs) {
            const centerWord = vocabulary[pair[0]];
            const contextWord = vocabulary[pair[1]];
            console.log(`Center word: ${centerWord}, Context word: ${contextWord}`);
        }
    }

    getSubsamplingProbability(wordIndex) {
        const totalWords = Object.keys(this.wordVectors).length;
        const wordFrequency = this.wordFrequencies[wordIndex] ?? 0;
        return 1 - Math.sqrt(this.subsamplingThreshold / (wordFrequency / totalWords));
    }

    getWordVector(wordIndex) {
        return this.wordVectors[wordIndex];
    }

    getWordIndex(word) {
        const lowercaseWord = word.toLowerCase();
        const cleanedWord = lowercaseWord.replace(/[^a-z0-9]+/gi, '');
        return this.wordToIndex[cleanedWord];
    }
}


// Example usage
const vocabulary = ['The', 'quick', 'brown', 'fox', 'jumps', 'over', 'the', 'lazy', 'dog'];
const contextWindowSize = 2;
const embeddingSize = 50;
const epochs = 100;
const initialLearningRate = 0.01;

const model = new BuildHuffmanGramModel(embeddingSize);
const trainingPairs = model.generateTrainingPairs(vocabulary, contextWindowSize);
model.train(trainingPairs, epochs, initialLearningRate);

const word1 = 'quick';
const word2 = 'fox';
const word1Index = model.getWordIndex(word1);
const word2Index = model.getWordIndex(word2);
const similarity = model.evaluateSimilarity(word1Index, word2Index);
console.log(`Similarity between "${word1}" and "${word2}": ${similarity}`);
