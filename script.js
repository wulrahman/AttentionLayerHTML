document.addEventListener('DOMContentLoaded', function() {
    const inputStrings = [
        "what is the capital of France?",
        "how tall is the Eiffel Tower?",
        "what is the population of Tokyo?",
        "who is the president of the United States?",
        "what is the currency of Japan?",
        "how many time zones are there in the world?",
        "what is the largest country in the world?",
        "who painted the Mona Lisa?",
        "what is the square root of 16?",
        "what is the capital of Australia?",
        "how many planets are there in our solar system?",
        "who wrote the book 'To Kill a Mockingbird'?",
        "what is the largest ocean in the world?",
        "what is the national animal of India?",
        "how many bones are there in the human body?",
        "who discovered electricity?",
        "what is the chemical symbol for gold?",
        "what is the largest desert in the world?",
        "who is the author of the Harry Potter series?",
        "what is the speed of light in a vacuum?"
    ];

    const targetStrings = [
        "Paris is the capital of France. It is known as the city of love.",
        "The Eiffel Tower is 330 meters tall. It was built in 1889.",
        "The population of Tokyo is approximately 14 million people.",
        "The current president of the United States is Joe Biden.",
        "The currency of Japan is the Japanese yen.",
        "There are 24 time zones in the world.",
        "The largest country in the world by land area is Russia.",
        "The Mona Lisa was painted by Leonardo da Vinci.",
        "The square root of 16 is 4.",
        "The capital of Australia is Canberra.",
        "There are 8 planets in our solar system.",
        "The book 'To Kill a Mockingbird' was written by Harper Lee.",
        "The largest ocean in the world is the Pacific Ocean.",
        "The national animal of India is the Bengal tiger.",
        "There are 206 bones in the human body.",
        "Electricity was discovered by Benjamin Franklin.",
        "The chemical symbol for gold is Au.",
        "The largest desert in the world is the Sahara Desert.",
        "The author of the Harry Potter series is J.K. Rowling.",
        "The speed of light in a vacuum is approximately 299,792,458 meters per second."
    ];

    const additionalInputStrings = [
        "what is the capital of China?",
        "how deep is the Mariana Trench?",
        "what is the population of India?",
        "who is the Prime Minister of the United Kingdom?",
        "what is the largest mountain in the world?",
        "how many continents are there?",
        "what is the official language of Brazil?",
        "who invented the telephone?",
        "what is the chemical symbol for oxygen?",
        "what is the largest waterfall in the world?",
        "who is the author of 'Pride and Prejudice'?",
        "what is the national bird of the United States?",
        "how many teeth does an adult human have?",
        "who discovered gravity?",
        "what is the atomic number of carbon?",
        "what is the highest temperature ever recorded?",
        "who is the founder of Microsoft?",
        "what is the largest lake in Africa?",
        "who is the author of '1984'?",
        "what is the distance from the Earth to the Moon?"
    ];

    const additionalTargetStrings = [
        "The capital of China is Beijing.",
        "The Mariana Trench is approximately 11,034 meters deep.",
        "The population of India is approximately 1.3 billion people.",
        "The current Prime Minister of the United Kingdom is Boris Johnson.",
        "The largest mountain in the world is Mount Everest.",
        "There are 7 continents in the world.",
        "The official language of Brazil is Portuguese.",
        "The telephone was invented by Alexander Graham Bell.",
        "The chemical symbol for oxygen is O.",
        "The largest waterfall in the world is Angel Falls in Venezuela.",
        "The author of 'Pride and Prejudice' is Jane Austen.",
        "The national bird of the United States is the bald eagle.",
        "An adult human has 32 teeth.",
        "Gravity was discovered by Sir Isaac Newton.",
        "The atomic number of carbon is 6.",
        "The highest temperature ever recorded was 56.7 degrees Celsius (134 degrees Fahrenheit) in Death Valley, California, USA.",
        "Microsoft was founded by Bill Gates and Paul Allen.",
        "The largest lake in Africa is Lake Victoria.",
        "The author of '1984' is George Orwell.",
        "The distance from the Earth to the Moon is approximately 384,400 kilometers (238,900 miles)."
    ];

    inputStrings.push(...additionalInputStrings);
    targetStrings.push(...additionalTargetStrings);

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
    var epoch = 100000;

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
                const output = attention.forward(query, query.positionEncode(), value);
                if(i == epoch) {
                    attention.visualizeAttentions(query, query.positionEncode(), value);
                }
                const currentLoss = output.meanSquaredError(target);
                if (isNaN(currentLoss)) {
                    console.log("Training stopped due to NaN loss.");
                    return losses;
                }
                loss += currentLoss;
                totalLoss.add(target.subtract(output));
            }

            totalLoss.divideScaler(inputStrings.length);
            learningRates.push(learningRate);
            attention.backward(totalLoss, learningRate);


            // if (losses.length > 0 && loss > losses[losses.length - 1]) {
            //     learningRate *= 0.995;
            // }

            losses.push(loss);
            console.log(`Epoch ${i} Loss: ${loss}`);
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
        const output = attention.forward(query,  query.positionEncode(), value);    
        return indexedDictionaryModel.matrix2String(output);
    }


    const inputElement = document.getElementById('inputElement');
    const response = document.getElementById('response');


    inputElement.addEventListener('input', function(event) {
        const userInput = event.target.value;
        response.innerText = generateResponse(userInput);
        console.log(response);
    });
}); 
