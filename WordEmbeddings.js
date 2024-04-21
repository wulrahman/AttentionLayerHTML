class WordEmbeddings {
    constructor(windowSize, embeddingSize, learningRate, numIterations) {
        this.windowSize = windowSize;
        this.embeddingSize = embeddingSize;
        this.learningRate = learningRate;
        this.numIterations = numIterations;
        this.vocabulary = {};
        this.embeddings = {};
    }

    preprocessTextData(text) {
        const cleanedCorpus = text.toLowerCase().replace(/[^\w\s]/gi, '').split(' ');
        cleanedCorpus.forEach(word => {
            if (!this.vocabulary[word]) {
                this.vocabulary[word] = Object.keys(this.vocabulary).length;
            }
        });
        return cleanedCorpus;
    }

    generateTrainingData(text) {
        const dataArray = this.preprocessTextData(text);
        const trainingData = [];
        for (let i = 0; i < dataArray.length; i++) {
            const targetWord = dataArray[i];
            for (let j = Math.max(0, i - this.windowSize); j < Math.min(dataArray.length, i + this.windowSize); j++) {
                if (i !== j) {
                    const contextWord = dataArray[j];
                    trainingData.push([targetWord, contextWord]);
                }
            }
        }
        return trainingData;
    }

    initializeWordEmbeddings(word) {
        this.embeddings[word] = new Array(this.embeddingSize).fill(1/this.embeddingSize);
    }

    cosineSimilarity(embedding1, embedding2) {
        let dotProduct = 0;
        let norm1 = 0;
        let norm2 = 0;
        for (let i = 0; i < embedding1.length; i++) {
            dotProduct += embedding1[i] * embedding2[i];
            norm1 += embedding1[i] * embedding1[i];
            norm2 += embedding2[i] * embedding2[i];
        }
        return dotProduct / (Math.sqrt(norm1) * Math.sqrt(norm2));
    }

    wordEmbedding(word) {
        return this.embeddings[word] || new Array(this.embeddingSize).fill(0);
    }

    textToMatrix(text, row, col) { 
        const inputsArray = text.toLowerCase().replace(/[^\w\s]/gi, '').split(' ');
        const embedding = new Matrix(row, col);
        inputsArray.forEach((char, index) => {
            const index_value = this.wordEmbedding(char);
            for (let i = 0; i < index_value.length; i++) {
                embedding.set(index, i, parseFloat(index_value[i]));
            }
        })
        return embedding;
    }

    findSimilarWords(targetWord, numSimilarWords) {
        const targetEmbedding = this.embeddings[targetWord];
        const similarWords = Object.keys(this.embeddings).map(word => {
            return {
                word: word,
                similarity: this.cosineSimilarity(targetEmbedding, this.embeddings[word])
            };
        });
        similarWords.sort((a, b) => b.similarity - a.similarity);
        return similarWords.slice(0, numSimilarWords);
    }
    
    embeddingtoWord(embedding) {
        let mostSimilarWord = null;
        let maxSimilarity = -Infinity;
        Object.keys(this.embeddings).forEach(word => {
            const similarity = this.cosineSimilarity(embedding, this.embeddings[word]);
            if (similarity > maxSimilarity) {
                mostSimilarWord = word;
                maxSimilarity = similarity;
            }
        });
        return mostSimilarWord;
    }

    embeddingtoText(embedding) {
        const text = [];
        for (let i = 0; i < embedding.rows; i++) {
            text.push(this.embeddingtoWord(embedding.data[i]));
        }
        return text;
    }

    trainWordEmbeddings(text) {
        const trainingData = this.generateTrainingData(text);
        for (let iter = 0; iter < this.numIterations; iter++) {
            trainingData.forEach(([targetWord, contextWord]) => {
                if (!this.embeddings[targetWord]) {
                    this.initializeWordEmbeddings(targetWord);
                }
                if (!this.embeddings[contextWord]) {
                    this.initializeWordEmbeddings(contextWord);
                }

                const targetEmbedding = this.embeddings[targetWord];
                const contextEmbedding = this.embeddings[contextWord];
                // Update target embedding
                for (let i = 0; i < this.embeddingSize; i++) {
                    targetEmbedding[i] -= this.learningRate * contextEmbedding[i];
                }

                // Update context embedding
                for (let i = 0; i < this.embeddingSize; i++) {
                    contextEmbedding[i] -= this.learningRate * targetEmbedding[i];
                }

                this.embeddings[targetWord] = targetEmbedding;
            });
        }
    }

    getWordEmbeddings() {
        return this.embeddings;
    }
}