import numpy as np
class CrossEntropyLoss:
    def forward(self, inputs, targets):
        samples = len(inputs)
        clipped_inputs = np.clip(inputs, 1e-7, 1 - 1e-7) #avoiding vlaus beocming to small,prevent log(0) and ensure numerical stability
        correct_confidences = clipped_inputs[range(samples), targets]
        negative_log=-np.log(correct_confidences)
        cross_entropy_loss=np.mean(negative_log)
        return cross_entropy_loss

    def backward(self,prob_values, targets):
        samples = len(prob_values)
        dvalues_copy = prob_values.copy()
        dvalues_copy[range(samples), targets] -= 1
        return dvalues_copy / samples #output/samples