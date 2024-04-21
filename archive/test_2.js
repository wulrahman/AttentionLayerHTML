function BuildHuffmanGram(data) {
    // Step 1: Count the frequency of each character in the data
    const frequencyMap = {};
    for (let i = 0; i < data.length; i++) {
        const char = data[i];
        if (frequencyMap[char]) {
            frequencyMap[char]++;
        } else {
            frequencyMap[char] = 1;
        }
    }

    // Step 2: Create a priority queue of nodes
    const queue = [];
    for (const char in frequencyMap) {
        queue.push({ char, frequency: frequencyMap[char] });
    }
    queue.sort((a, b) => a.frequency - b.frequency);

    // Step 3: Build the Huffman tree
    while (queue.length > 1) {
        const left = queue.shift();
        const right = queue.shift();
        const newNode = {
            char: null,
            frequency: left.frequency + right.frequency,
            leftChild: left,
            rightChild: right,
        };
        queue.push(newNode);
        queue.sort((a, b) => a.frequency - b.frequency);
    }

    // Step 4: Return the root of the Huffman tree
    return queue[0];
}

function EncodeHuffmanGram(data) {
    const root = BuildHuffmanGram(data);
    const encodingMap = {};
    const encode = (node, prefix) => {
        if (node.char) {
            encodingMap[node.char] = prefix;
        } else {
            encode(node.leftChild, prefix + '0');
            encode(node.rightChild, prefix + '1');
        }
    };
    encode(root, '');

    let encodedData = '';
    for (let i = 0; i < data.length; i++) {
        encodedData += encodingMap[data[i]];
    }

    return { encodedData, encodingMap };
}

function DecodeHuffmanGram(encodedData, encodingMap) {
    let decodedData = '';
    let currentChar = '';
    for (let i = 0; i < encodedData.length; i++) {
        currentChar += encodedData[i];
        if (encodingMap[currentChar]) {
            decodedData += encodingMap[currentChar];
            currentChar = '';
        }
    }

    return decodedData;
}

// Example usage
const data = 'hello world';
const { encodedData, encodingMap } = EncodeHuffmanGram(data);
console.log('Encoded data:', encodedData);
