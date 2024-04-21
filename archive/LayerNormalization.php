
<?php

class LayerNormalization {

    public function __construct() {
    }

    public function normalize($input) {
        // Normalize the input using layer normalization
        $mean = $this->compute_mean($input);
        $variance = $this->compute_variance($input, $mean);
        $normalized_input = $this->apply_normalization($input, $mean, $variance);
        return $normalized_input;
    }

    private function compute_mean($input) {
        // Compute the mean of the input
        $rows = count($input);
        $cols = count($input[0]);
        $mean = [];

        for ($j = 0; $j < $cols; $j++) {
            $sum = 0;
            for ($i = 0; $i < $rows; $i++) {
                $sum += $input[$i][$j];
            }
            $mean[$j] = $sum / $rows;
        }

        return $mean;
    }

    private function compute_variance($input, $mean) {
        // Compute the variance of the input
        $rows = count($input);
        $cols = count($input[0]);
        $variance = [];

        for ($j = 0; $j < $cols; $j++) {
            $sum = 0;
            for ($i = 0; $i < $rows; $i++) {
                $sum += pow($input[$i][$j] - $mean[$j], 2);
            }
            $variance[$j] = $sum / $rows;
        }

        return $variance;
    }

    private function apply_normalization($input, $mean, $variance) {
        // Apply layer normalization to the input
        $rows = count($input);
        $cols = count($input[0]);
        $normalized_input = [];

        for ($i = 0; $i < $rows; $i++) {
            for ($j = 0; $j < $cols; $j++) {
                $normalized_input[$i][$j] = ($input[$i][$j] - $mean[$j]) / sqrt($variance[$j] + 1e-8);
            }
        }

        return $normalized_input;
    }

    public function backward($gradient, $learningRate) {
        // Backward pass through layer normalization
        $rows = count($gradient);
        $cols = count($gradient[0]);
        $new_gradient = [];

        for ($i = 0; $i < $rows; $i++) {
            for ($j = 0; $j < $cols; $j++) {
                $new_gradient[$i][$j] = $gradient[$i][$j] * $learningRate;
            }
        }

        return $new_gradient;
    }

    public function update($input, $gradient, $learningRate) {
        // Update the input using the gradient and learning rate
        $rows = count($input);
        $cols = count($input[0]);
        $new_input = [];

        for ($i = 0; $i < $rows; $i++) {
            for ($j = 0; $j < $cols; $j++) {
                $new_input[$i][$j] = $input[$i][$j] - $gradient[$i][$j] * $learningRate;
            }
        }

        return $new_input;
    }

    public function forward($input) {
        // Forward pass through layer normalization
        return $this->normalize($input);
    }

    public function train($input, $gradient, $learningRate) {
        // Backward pass through layer normalization
        $new_gradient = $this->backward($gradient, $learningRate);

        // Update the input using the gradient
        $new_input = $this->update($input, $new_gradient, $learningRate);

        return $new_input;
    }
}

// Example usage
$layerNormalization = new LayerNormalization();
$input = [[1, 2, 3], [4, 5, 6], [7, 8, 9]];
$normalized_input = $layerNormalization->normalize($input);

print("Normalized input:\n");
foreach ($normalized_input as $row) {
    print(implode(", ", $row) . "\n");
}

