class Word2Vec {
    constructor(vocabularySize, embeddingSize) {
        this.vocabularySize = vocabularySize;
        this.embeddingSize = embeddingSize;

        // Initialize the word embeddings
        this.wordEmbeddings = [];
        for (let i = 0; i < vocabularySize; i++) {
            this.wordEmbeddings[i] = Matrix.matrixValue(1/this.vocabularySize, 1, embeddingSize);
        }
    }

    train(sentences, windowSize, learningRate, numIterations) {
        for (let iteration = 0; iteration < numIterations; iteration++) {
            for (let sentence of sentences) {
                for (let i = 0; i < sentence.length; i++) {
                    const centerWord = sentence[i];

                    // Get the context words within the window size
                    const contextWords = sentence.slice(Math.max(0, i - windowSize), i).concat(sentence.slice(i + 1, i + windowSize + 1));

                    // Update the word embeddings for the center word and context words
                    for (let contextWord of contextWords) {
                        const centerEmbedding = this.wordEmbeddings[centerWord];
                        const contextEmbedding = this.wordEmbeddings[contextWord];

                        if(centerEmbedding != undefined && contextEmbedding != undefined) {
                        // Calculate the dot product of the center and context embeddings
                            const dotProduct = Matrix.dot(centerEmbedding, contextEmbedding.transpose());

                            // Calculate the gradients for the center and context embeddings
                            const gradientsCenter = contextEmbedding.multiplyScaler(dotProduct.data[0][0]);
                            const gradientsContext = centerEmbedding.multiplyScaler(dotProduct.data[0][0]);

                            // Update the center and context embeddings
                            this.wordEmbeddings[centerWord].add(gradientsCenter.multiplyScaler(learningRate));
                            this.wordEmbeddings[contextWord].add(gradientsContext.multiplyScaler(learningRate));

                        }
                    }
                }
            }
        }
    }

    getWordEmbedding(word) {
        return this.wordEmbeddings[word];
    }
}




// // Define the vocabulary size and embedding size
// const vocabularySize = 3;
// const embeddingSize = 3;

// // Create an instance of the Word2Vec model
// const word2vec = new Word2Vec(vocabularySize, embeddingSize);

// // Define the sentences and other parameters
// const sentences = [
//     [0, 1, 2, 3, 4], // Sentence 1
//     [5, 6, 7, 8, 9], // Sentence 2
//     [10, 11, 12, 13, 14] // Sentence 3
// ];
// const windowSize = 2;
// const learningRate = 0.1;
// const numIterations = 100;

// // Train the Word2Vec model
// word2vec.train(sentences, windowSize, learningRate, numIterations);

// // Get the word embedding for a specific word
// const word = 2;
// const embedding = word2vec.getWordEmbedding(word);
// console.log(`Word embedding for word ${word}:`, embedding);