class Word2Vec {
    constructor() {
        this.word2vec = {};
    }
    
    // Train the model
    train(corpus, windowSize) {

        // Create a set of unique words
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

            for (let otherword of uniqueWords) {
                this.word2vec[word][otherword] = 0;
            }
        }

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

    
    stringToMatrix(words, row, col) {
        const array = [];
        const embedding = new Matrix(row, col);
        words.toLowerCase().replace(/[^\w\s]/gi, '').split(" ").forEach((char, index) => {
            const index_value = word2vec.calculateWordEmbedding(char);
            for (let i = 0; i < index_value.length; i++) {
                embedding.set(index, i, parseFloat(index_value[i]));
            }
        })
        return embedding;
    }

    getWordFrequency() {
        let wordFrequency = {};

        for (const word in this.word2vec) {
            let contextWords = Object.keys(this.word2vec[word]);
            let frequency = 0;
            for (let i = 0; i < contextWords.length; i++) {
                let contextWord = contextWords[i];
                frequency += this.word2vec[word][contextWord];
            }
            wordFrequency[word] = frequency;
        }

        return wordFrequency;
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

// // Example usage
// const word2vec1 = new Word2Vector();

// word2vec1.train([
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
// // word2vec.print();
// // console.log(word2vec.predict("HTML")); // provide
// // console.log(word2vec.predict("provide")); // feedback
// // console.log(word2vec.predict("feedback")); // and
// // console.log(word2vec.predict("and")); // suggestions
// // console.log(word2vec.predict("suggestions")); // during
// // console.log(word2vec.predict("during")); // code
// // console.log(word2vec.predict("code")); // reviews
// // console.log(word2vec.predict("reviews")); // I

// // console.log(word2vec.calculateWordEmbedding("code")); // 1.5
// console.log(word2vec1.consineSimilarity(word2vec1.calculateWordEmbedding("react"), word2vec1.calculateWordEmbedding("php")));
// console.log(word2vec1.consineSimilarity(word2vec1.calculateWordEmbedding("machine"), word2vec1.calculateWordEmbedding("artificial")));

// // console.log(word2vec.findClosestMatch("a")); // I
// // console.log(word2vec.getSimilarWords("PHP", 5)); // ["reviews", "during", "I", "provide", "feedback"]