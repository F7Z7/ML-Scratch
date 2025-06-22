import numpy as np
class CrossEntropyLoss:
    def forward(self, inputs, targets):
        samples = len(inputs)
        clipped_inputs = np.clip(inputs, 1e-7, 1 - 1e-7) #avoiding vlaus beocming to small,prevent log(0) and ensure numerical stability
        correct_confidences = clipped_inputs[range(samples), targets]
        negative_log=-np.log(correct_confidences)
        cross_entropy_loss=-np.sum(negative_log)
        return cross_entropy_loss

    def backward(self,predictions, targets):
        samples = predictions.shape[0]
        self.dinputs = (predictions - targets) / samples  # derivative of softmax + cross-entropy
        return self.dinputs