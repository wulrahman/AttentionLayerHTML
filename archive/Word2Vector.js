class Word2Vector {
    constructor() {
        this.word2vec = {};
    }
    
    // Train the model
    train(corpus, windowSize) {

        let uniqueWords = new Set();
        for (let i = 0; i < corpus.length; i++) {
            let words = corpus[i].toLowerCase().replace(/[^\w\s]/gi, '').split(' ')
            for (let j = 0; j < words.length; j++) {
                uniqueWords.add(words[j]);
            }
        }

        // Initialize the word count
        for (let word of uniqueWords) {

            if (this.word2vec[word] === undefined) {
                this.word2vec[word] = {};
            }
            
            this.word2vec[word][word] = 0;

            for (let otherword of uniqueWords) {
                this.word2vec[word][otherword] = 0;
            }
        }

        console.log(this.word2vec['is']);

        for (let i = 0; i < corpus.length; i++) {
            let words = corpus[i].toLowerCase().replace(/[^\w\s]/gi, '').split(' ');
            for (let j = 0; j < words.length; j++) {
                let word = words[j];
                for (let k = j - windowSize; k <= j + windowSize; k++) {
                    if (k !== j && k >= 0 && k < words.length) {
                        let contextWord = words[k];
                        this.word2vec[word][contextWord]++;
                    }
                }
            }
        }
    }

    
    calculateCentroid() {
        let centroid = {};
        let count = 0;
        for (const word in this.word2vec) {
            for (const contextWord in this.word2vec[word]) {
                if (centroid[contextWord] === undefined) {
                    centroid[contextWord] = 0;
                }
                centroid[contextWord] += this.word2vec[word][contextWord];
                count++;
            }
        }
        for (const word in centroid) {
            centroid[word] /= count;
        }
        return centroid;
    }

    calculateWordEmbedding(word) {
        let centroid = this.calculateCentroid();

        let wordVector = this.word2vec[word];
        let wordEmbedding = {};
        for (const contextWord in wordVector) {
            if (centroid[contextWord] !== undefined) {
                wordEmbedding[contextWord] = wordVector[contextWord] - centroid[contextWord];
            }
        }
        return wordEmbedding;
    }


    stringToMatrix(text, row, col) {
        const inputsArray = text.toLowerCase().replace(/[^\w\s]/gi, '').split(' ');
        const embedding = new Matrix(row, col);
        inputsArray.forEach((char, index) => {
            const index_value = this.getWordEmbedding(char);
            for (let i = 0; i < index_value.length; i++) {
                embedding.set(index, i, parseFloat(index_value[i]));
            }
        })
        return embedding;
    }


    // Get the word vector representation
    getWordVector(word) {
        let wordVector = [];
        let nextWords = this.word2vec[word];
        for (let key in nextWords) {
            wordVector.push(nextWords[key]);
        }

        if (wordVector.length === 0) {
            wordVector.push(0);
        }
        return wordVector;
    }

    getSimilarWords(word, numWords) {
        let similarWords = [];
        let nextWords = this.word2vec[word];
        let sortedWords = Object.keys(nextWords).sort((a, b) => nextWords[b] - nextWords[a]);
        for (let i = 0; i < numWords; i++) {
            similarWords.push(sortedWords[i]);
        }
        return similarWords;
    }


    normalizeWord2Vec() {
        let minCount = Infinity;
        let maxCount = -Infinity;

        // Find the minimum and maximum counts
        for (const word in this.word2vec) {
            for (const contextWord in this.word2vec[word]) {
                const count = this.word2vec[word][contextWord];
                if (count < minCount) {
                    minCount = count;
                }
                if (count > maxCount) {
                    maxCount = count;
                }
            }
        }

        // Normalize the counts
        for (const word in this.word2vec) {
            for (const contextWord in this.word2vec[word]) {
                this.word2vec[word][contextWord] = (this.word2vec[word][contextWord] - minCount) / (maxCount - minCount);
            }
        }
    }

    consineSimilarity(vectorA, vectorB) {
        let dotProduct = 0;
        let normA = 0;
        let normB = 0;

        vectorA = Object.values(vectorA);
        vectorB = Object.values(vectorB);
        for (let i = 0; i < vectorA.length; i++) {
            dotProduct += vectorA[i] * vectorB[i];
            normA += Math.pow(vectorA[i], 2);
            normB += Math.pow(vectorB[i], 2);
        }
        return dotProduct / (Math.sqrt(normA) * Math.sqrt(normB));
    }

    getWordEmbedding(word) {
        let wordVector = this.getWordVector(word);
        let wordEmbedding = wordVector.reduce((acc, val) => acc + val, 0) / wordVector.length;
        return wordEmbedding;
    }

    findClosestMatch(query) {
        let closestMatch = '';
        let minDistance = Infinity;
        const queryEmbedding = this.calculateWordEmbedding(query);

        for (const key in this.word2vec) {
            const keyEmbedding = this.calculateWordEmbedding(key);
            
            if (queryEmbedding && keyEmbedding) {
                const distance = Math.abs(this.consineSimilarity(queryEmbedding, keyEmbedding));
                if (distance < minDistance) {
                    closestMatch = key;
                    minDistance = distance;
                }
            }
        }
        return closestMatch;
    }

    // Print the model
    print() {
        console.log(this.word2vec);
    }
}

// // Example usage
// const word2vec = new Word2Vector();
// word2vec.train([
//     "I provide feedback and suggestions during code reviews.",
//     "I don't have a favorite testing framework, I can work with any.",
//     "JavaScript is a popular programming language for web development.",
//     "Machine learning is a subset of artificial intelligence.",
//     "Version control systems like Git are essential for collaborative software development.",
//     "Python is widely used in data science and machine learning.",
//     "CSS is used for styling web pages.",
//     "HTML is the standard markup language for creating web pages.",
//     "React is a JavaScript library for building user interfaces.",
//     "Node.js allows you to run JavaScript on the server side.",
//     "Angular is a popular framework for building web applications.",
//     "Java is a widely used programming language for enterprise applications.",
//     "C# is a programming language developed by Microsoft.",
//     "Ruby is known for its simplicity and productivity.",
//     "PHP is a server-side scripting language used for web development.",
//     "Swift is a programming language used for iOS app development.",
//     "Go is a statically typed, compiled programming language.",
//     "Rust is designed for systems programming with a focus on safety and concurrency.",
//     "TypeScript is a typed superset of JavaScript that compiles to plain JavaScript.",
//     "Kotlin is a modern programming language that runs on the Java Virtual Machine."
// ], 2);
// word2vec.print();
// console.log(word2vec.predict("HTML")); // provide
// console.log(word2vec.predict("provide")); // feedback
// console.log(word2vec.predict("feedback")); // and
// console.log(word2vec.predict("and")); // suggestions
// console.log(word2vec.predict("suggestions")); // during
// console.log(word2vec.predict("during")); // code
// console.log(word2vec.predict("code")); // reviews
// console.log(word2vec.predict("reviews")); // I

// console.log(word2vec.getWordEmbedding("a")); // 1.5
// console.log(word2vec.findClosestMatch("a")); // I
// console.log(word2vec.getSimilarWords("PHP", 5)); // ["reviews", "during", "I", "provide", "feedback"]