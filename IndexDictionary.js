class IndexDictionary {

    constructor() {
        this.dictionary = { "": 0 };
        this.index = 1;
    }

    addWord(word) {
        const words = word.toLowerCase().replace(/[^\w\s]/gi, '').split(" ");
        words.forEach(word => {
            if (!this.dictionary.hasOwnProperty(word)) {
                this.dictionary[word] = this.index;
                this.index++;
            }
        });
    }

    normalizeDistribution() {
        const distribution = Object.values(this.dictionary);;
        this.total = distribution.reduce((acc, val) => acc + val, 0);
        const normalizedData = distribution.map(val => val / this.total);
        let i = 0;
        for (const key in this.dictionary) {
            this.dictionary[key] = normalizedData[i];
            i++;
        }
        return this.dictionary;
    }

    normalizeMinMax() {
        const distribution = Object.values(this.dictionary);
        const min = Math.min(...distribution);
        const max = Math.max(...distribution);
        const normalizedData = distribution.map(val => (val - min) / (max - min));
        let i = 0;
        for (const key in this.dictionary) {
            this.dictionary[key] = normalizedData[i];
            i++;
        }
        return this.dictionary;
    }


    stringToMatrix(text, row, col) {

        // const inputsArray = text.toLowerCase().replace(/[^\w\s]/gi, '').split(' ');
        // const embedding = new Matrix(row, col);
        // inputsArray.forEach((char, index) => {
        //     embedding.set(0, index, this.getIndex(char));
        // })
        // return embedding;
    
        const array = [];
        text.toLowerCase().replace(/[^\w\s]/gi, '').split(" ").forEach((word, i) => {
            array.push(this.getIndex(word));
        });
        const embedding = Matrix.arraytoMatrix(array, row, col, 0);
        return embedding;
    }

    getWord(index) {
        const word = Object.keys(this.dictionary).find(key => this.dictionary[key] === index);
        if (word) {
            return word;
        }
        return '';
    }

    findClosestMatch(query) {
        let closestMatch = '';
        let minDistance = Infinity;
        for (const key in this.dictionary) {
            const distance = this.calculateDistanceValue(query, this.dictionary[key]);
            if (distance < minDistance) {
                closestMatch = key;
                minDistance = distance;
            }
        }
        return closestMatch;
    }

    matrix2String(matrix) {
        const array = matrix.flatten().data[0];
        return array.map(val => this.findClosestMatch(val)).join(" ");
    }

    calculateDistance(query, word) {
        let distance = 0;
        for (let i = 0; i < query.length; i++) {
            distance += Math.abs(query.charCodeAt(i) - word.charCodeAt(i));
        }
        return distance;
    }

    
    calculateDistanceValue(query, word) {
        return  Math.abs(query - word);
    }


    getIndex(word) {
        return this.dictionary.hasOwnProperty(word) ? this.dictionary[word] : 0;
    }

    getDictionary() {
        return this.dictionary;
    }


}
