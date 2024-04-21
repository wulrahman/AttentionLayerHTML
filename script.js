document.addEventListener('DOMContentLoaded', function() {
    const inputStrings = [
        "what is your name?",
        "who is your father?"
    ];

    const targetStrings = [
        "I am a chatbot.",
        "my father is a waheed.",
    ];

    const uniqueWords = new Set();

    inputStrings.forEach(inputString => {
        const words = inputString.toLowerCase().replace(/[^\w\s]/gi, '').split(' ');
        words.forEach(word => uniqueWords.add(word));
    });

    targetStrings.forEach(targetString => {
        const words = targetString.toLowerCase().replace(/[^\w\s]/gi, '').split(' ');
        words.forEach(word => uniqueWords.add(word));
    });

    function countWords(str) {
        const words = str.toLowerCase().replace(/[^\w\s]/gi, '').split(' ');
        return words.length;
    }

    function averageWordsPerSentence(str) {
        const sentences = str.split(/[.!?]/);
        const totalWords = sentences.reduce((total, sentence) => total + countWords(sentence), 0);
        return totalWords / sentences.length;
    }

    function averageWordsPerInput(inputStrings) {
        const totalWords = inputStrings.reduce((total, inputString) => total + countWords(inputString), 0);
        return totalWords / inputStrings.length;
    }

    function maximumWordsPerSentence(str) {
        const sentences = str.split(/[.!?]/);
        const wordsPerSentence = sentences.map(sentence => countWords(sentence));
        return Math.max(...wordsPerSentence);
    }

    function maximumWordsPerInput(inputStrings) {
        const wordsPerInput = inputStrings.map(inputString => countWords(inputString));
        return Math.max(...wordsPerInput);
    }

    const maxWordsPerInput = maximumWordsPerInput(inputStrings);
    const maxWordsPerTarget = maximumWordsPerInput(targetStrings);
    const maxWordsPerInputAndTarget = Math.max(maxWordsPerInput, maxWordsPerTarget);
    
    var learningRate = 1e-4;
    var epoch = 1000000;

    const dModel = maxWordsPerInputAndTarget;
    const numHeads =  maxWordsPerInputAndTarget;

    const huffmanModel = new BuildHuffmanGramModel();
    const indexedDictionaryModel = new IndexDictionary();
    const wordEmbeddings = new WordEmbeddings(maxWordsPerInputAndTarget, dModel, 1e-5, 1000);
    const attention = new MultiheadAttention(dModel, numHeads);

    inputStrings.forEach((inputString, index) => {
        indexedDictionaryModel.addWord(inputString);
        indexedDictionaryModel.addWord(targetStrings[index]);
        const inputStringclean = inputString.toLowerCase().replace(/[^\w\s]/gi, '')
        const targetString = targetStrings[index].toLowerCase().replace(/[^\w\s]/gi, '')

        huffmanModel.buildFrequencyMap(targetString);
        wordEmbeddings.trainWordEmbeddings(targetString);
        huffmanModel.buildFrequencyMap(inputStringclean);
        wordEmbeddings.trainWordEmbeddings( inputStringclean);

    });


    console.log(wordEmbeddings.getWordEmbeddings())


    indexedDictionaryModel.normalizeMinMax();
    huffmanModel.buildHuffmanTree();
    huffmanModel.buildHuffmanCodeMap();

    function trainModel(learningRate, inputStrings, targetStrings, epoch, dModel, numHeads) {

        const losses = [];

        const learningRates = [];

        for (let i = 0; i <= epoch; i++) {
            const totalLoss = new Matrix(dModel, dModel);
 
            var loss = 0;
            for (let j = 0; j < inputStrings.length; j++) {
                const inputString = inputStrings[j].toLowerCase().replace(/[^\w\s]/gi, '');
                const query = wordEmbeddings.textToMatrix(inputString, dModel, dModel);
                const key = huffmanModel.textToMatrix(inputString, dModel, dModel, 0);
                const value = indexedDictionaryModel.stringToMatrix(inputString, dModel, dModel);

                const targetString = targetStrings[j].toLowerCase().replace(/[^\w\s]/gi, '');
                const target = indexedDictionaryModel.stringToMatrix(targetString, dModel, dModel);

                // const target = wordEmbeddings.textToMatrix(targetString, dModel, dModel);
                const output = attention.forward(query, key, value, false);
                if(i == epoch) {
                    attention.visualizeAttentions(query, key, value);
                }

                attention.backward(target.subtract(output), learningRate);

                const currentLoss = output.meanSquaredError(target);
                if (isNaN(currentLoss)) {
                    console.log("Training stopped due to NaN loss.");
                    return losses;
                }
                loss += currentLoss;
                totalLoss.add(target.subtract(output));
            }

            if (i % 1000 === 0) {
                const randomIndex = Math.floor(Math.random() * inputStrings.length);
                const randomInput = inputStrings[randomIndex];
                const randomTarget = targetStrings[randomIndex];
                const generatedResponse = generateResponse(randomInput);
                console.log(`Generated response for input "${randomInput}": ${generatedResponse}`);
            }

            totalLoss.divideScaler(inputStrings.length);
            learningRates.push(learningRate);


            if (losses.length > 0 && loss > losses[losses.length - 1]) {
                learningRate *= 0.995;
            }

            if (i % 100 === 0) {
                console.log(`Epoch ${i} Loss: ${loss}`);
                losses.push(loss);

            }

        }

        return losses;
    }

    function processLosses(losses) {
        const minLoss = Math.min(...losses.filter(loss => !isNaN(loss)));
        const minLossIndex = losses.indexOf(minLoss);

        console.log(`Lowest loss of ${minLoss} at epoch ${minLossIndex} with a learning rate of ${learningRate}`);
        
        const canvas = document.getElementById('lossGraph');
        const maxLoss = Math.max(...losses);
        const scaledLosses = losses.map(loss => (loss - minLoss) / (maxLoss - minLoss) * canvas.clientHeight);

        scaledLosses.forEach((loss, i) => {
            plotLoss(i, loss, losses);
        });
    }

    const losses = trainModel(learningRate, inputStrings, targetStrings, epoch, dModel, numHeads);
    processLosses(losses);

    function plotLoss(i, loss, losses) {
        const canvas = document.getElementById('lossGraph');
        const context = canvas.getContext('2d');
        // Set the color for the point
        context.fillStyle = 'red'; // You can change the color as needed
        // Draw a point at the specified coordinates
        context.fillRect((i / losses.length) * canvas.clientWidth, loss, 1, 1); // Adjust the size of the point as needed
    }
    
    function generateResponse(input) {

        const target = input.toLowerCase().replace(/[^\w\s]/gi, '');
        const query = wordEmbeddings.textToMatrix(target, dModel, dModel);
        const value = indexedDictionaryModel.stringToMatrix(target, dModel, dModel);
        const key = huffmanModel.textToMatrix(target, dModel, dModel, 0)
        const output = attention.forward(query, key, value, false)    
        return indexedDictionaryModel.matrix2String(output)
        // return wordEmbeddings.embeddingtoText(output);
    }


    const inputElement = document.getElementById('inputElement');
    const response = document.getElementById('response');


    inputElement.addEventListener('input', function(event) {
        const userInput = event.target.value;
        response.innerText = generateResponse(userInput);
        console.log(response);
    });
}); 
