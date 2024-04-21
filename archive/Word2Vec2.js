class Word2Vec {
    constructor() {
        this.wordVectors = {};
    }

    train(sentences, vectorSize, windowSize, epochs) {
        const wordVectors = {};
        const wordCount = {};
        let sentenceCount = 0;

        // Function to build Huffman tree and assign codes to each word
        const buildHuffmanTree = () => {
            const words = Object.keys(wordCount);
            const vocab = words.map(word => ({ word, count: wordCount[word] }));
            vocab.sort((a, b) => b.count - a.count);

            const huffman = new BuildHuffmanGramModel2(vocab);
            huffman.initialize();
            return huffman;
        };

        const huffman = buildHuffmanTree();

        // Function to train the model
        const train = (word, contextWord) => {
            const wordVector = wordVectors[word] || Array(vectorSize).fill(0);
            const contextWordVector = wordVectors[contextWord] || Array(vectorSize).fill(0);

            const dotProduct = wordVector.reduce((acc, val, i) => acc + val * contextWordVector[i], 0);
            const sigmoid = 1 / (1 + Math.exp(-dotProduct));

            const error = sigmoid - 1;

            const wordGradient = wordVector.map(val => error * val);
            const contextWordGradient = contextWordVector.map(val => error * val);

            wordVectors[word] = wordVector.map((val, i) => val - contextWordGradient[i]);
            wordVectors[contextWord] = contextWordVector.map((val, i) => val - wordGradient[i]);
        };

        // Function to train the model on a sentence
        const trainSentence = sentence => {
            sentenceCount++;
            const words = sentence.toLowerCase().replace(/[^\w\s]/gi, '').split(' ');
            words.forEach((word, i) => {
                if (!wordCount[word]) {
                    wordCount[word] = 1;
                } else {
                    wordCount[word]++;
                }

                const start = Math.max(0, i - windowSize);
                const end = Math.min(words.length, i + windowSize + 1);

                for (let j = start; j < end; j++) {
                    if (i !== j) {
                        train(word, words[j]);
                    }
                }
            });
        };

        // Train the model on the sentences
        for (let i = 0; i < epochs; i++) {
            sentences.forEach(trainSentence);
        }

        this.wordVectors = wordVectors;

        console.log(wordVectors);
    }

    getWordEmbedding(word) {
        return this.wordVectors[word];
    }
}

class BuildHuffmanGramModel2 {

    constructor(vocab) {
        this.vocab = vocab;
        this.huffmanTree = null;
        this.huffmanCodeMap = {};

        this.buildFrequencyMap = this.buildFrequencyMap.bind(this);
        this.buildHuffmanTree = this.buildHuffmanTree.bind(this);
        this.buildHuffmanCodeMap = this.buildHuffmanCodeMap.bind(this);
        this.initialize = this.initialize.bind(this);

        this.initialize();

    }

    buildFrequencyMap() {
        this.frequencyMap = {};
        this.vocab.forEach(({ word, count }) => {
            this.frequencyMap[word] = count;
        });
    }

    buildHuffmanTree() {
        const nodes = this.vocab.map(({ word, count }) => new HuffmanNode1(word, count));

        while (nodes.length > 1) {
            nodes.sort((a, b) => a.frequency - b.frequency);
            const left = nodes.shift();
            const right = nodes.shift();
            const parent = new HuffmanNode1(null, left.frequency + right.frequency);
            parent.left = left;
            parent.right = right;
            nodes.push(parent);
        }

        this.huffmanTree = nodes[0];
    }

    buildHuffmanCodeMap() {
        const codeMap = {};

        const traverse = (node, code) => {
            if (node.isLeaf()) {
                codeMap[node.char] = code;
            } else {
                traverse(node.left, code + "0");
                traverse(node.right, code + "1");
            }
        };

        traverse(this.huffmanTree, "");

        this.huffmanCodeMap = codeMap;
    }

    initialize() {
        this.buildFrequencyMap();
        this.buildHuffmanTree();
        this.buildHuffmanCodeMap();
    }
}


class HuffmanNode1 {
    constructor(char, frequency) {
        this.char = char;
        this.frequency = frequency;
        this.left = null;
        this.right = null;
    }

    isLeaf() {
        return this.left === null && this.right === null;
    }
}

const word2vec = new Word2Vec();
const sentences = [
    "I provide feedback and suggestions during code reviews.",
    "I don't have a favorite testing framework, I can work with any."
];
word2vec.train(sentences, 100, 2, 1000);
console.log(word2vec.getWordEmbedding("feedback"));
console.log(word2vec.getWordEmbedding("testing"));
console.log(word2vec.getWordEmbedding("framework"));
console.log(word2vec.getWordEmbedding("code"));