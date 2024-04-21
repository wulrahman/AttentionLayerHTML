class BuildHuffmanGramModel {
    constructor() {
        this.frequencyMap = {};
        this.huffmanTree = null;
    }

    buildFrequencyMap(text) {
        text.toLowerCase().replace(/[^\w\s]/gi, '').split(' ').forEach(char => {
            if (this.frequencyMap[char]) {
                this.frequencyMap[char]++;
            } else {
                this.frequencyMap[char] = 1;
            }
        });
    }

    buildHuffmanTree() {
        const nodes = [];
        for (const char in this.frequencyMap) {
            nodes.push(new HuffmanNode(char, this.frequencyMap[char]));
        }

        while (nodes.length > 1) {
            nodes.sort((a, b) => a.frequency - b.frequency);
            const left = nodes.shift();
            const right = nodes.shift();
            const parent = new HuffmanNode(null, left.frequency + right.frequency);
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

        return codeMap;
    }

    encode(text) {
        let encoded = "";
        for (const char of text.split(" ")) {
            encoded += this.buildHuffmanCodeMap()[char];
        }
        return encoded;
    }

    textToMatrix(text, row, col) { 
        const inputsArray = text.toLowerCase().replace(/[^\w\s]/gi, '');
        const encoded = this.encode(inputsArray).split('');
        // const embedding = new Matrix(row, col);
        // inputsArray.forEach((char, index) => {
        //     const index_value = this.encode(char).split('');
        //     for (let i = 0; i < index_value.length; i++) {
        //         embedding.set(index, i, parseFloat(index_value[i]));
        //     }
        // })
        // return embedding;
        const embedding = Matrix.arraytoMatrix(encoded, row, col, 0);
        return embedding;
    }
    

    decode(encoded) {
        let decoded = "";
        let current = this.huffmanTree;
        for (const bit of encoded) {
            if (bit === "0") {
                current = current.left;
            } else {
                current = current.right;
            }

            if (current.isLeaf()) {
                decoded += current.char + " ";
                current = this.huffmanTree;
            }
        }

        return decoded;
    }

    buildHuffmanGramModel(text) {
        this.buildFrequencyMap(text);
        this.buildHuffmanTree();
        return this.encode(text);
    }
}

class HuffmanNode {
    constructor(char, frequency) {
        this.char = char;
        this.frequency = frequency;
        this.left = null;
        this.right = null;
    }

    isLeaf() {
        return !this.left && !this.right;
    }
}

// const huffmanModel = new BuildHuffmanGramModel();
// const inputStrings = [
//     "I provide feedback and suggestions during code reviews.",
//     "I don't have a favorite testing framework, I can work with any."
// ];

// let index = 1;
// inputStrings.forEach(inputString => {
//     console.log(huffmanModel.decode(huffmanModel.buildHuffmanGramModel(inputString)));
// });

// const targets = [
//     "I provide feedback and suggestions during code reviews.",
//     "I don't have a favorite testing framework, I can work with any."
// ];

// targets.forEach(targetString => {
//     huffmanModel.buildFrequencyMap(targetString);
// });

// huffmanModel.buildHuffmanTree();
// const indexedDictionary = huffmanModel.buildHuffmanCodeMap();

// const inputs = inputStrings.map(inputString => {
//     const query = new Matrix(dModel, dModel);
//     inputString.split(' ').forEach((char, index) => {
//         const index_value = indexedDictionary[char].split('');
//         for (let i = 0; i < index_value.length; i++) {
//             query.set(index, i, parseInt(index_value[i]));
//         }
//     });
//     return query;
// }
// );

// const outputTargets = targets.map(targetString => {
//     const target = new Matrix(dModel, dModel);
//     targetString.split(' ').forEach((char, index) => {
//         const index_value = indexedDictionary[char].split('');
//         if (index_value.length > 0) {
//             for (let i = 0; i < index_value.length; i++) {
//                 target.set(index, i, parseInt(index_value[i]));
//             }
//         }
//     });
//     return target;
// }
// );

// var loss;

// for (let i = 0; i < 100000; i++) {
    
//         loss = 0;
//         for (let j = 0; j < inputs.length; j++) {
//             const target = outputTargets[j];
//             const output = attention.forward(inputs[j], inputs[j], inputs[j]);
//             loss += output.meanSquaredError(target);
//             // Plot the loss in a graph
//             attention.backward(inputs[j], target, output, loss, 1e-8);
//         }
//     }
// // Path: Modules/multiLayerTransformModel/Attention/Attention.js
// // Compare this snippet from Modules/multiLayerTransformModel/Attention/script.js:
// //         "I provide feedback and suggestions during code reviews.",
