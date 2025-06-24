import numpy as np
class CrossEntropyLoss:
    def forward(self, inputs, targets):
        samples = len(inputs)
        clipped_inputs = np.clip(inputs, 1e-7, 1 - 1e-7) #avoiding vlaus beocming to small,prevent log(0) and ensure numerical stability

        if len(targets.shape) == 2:
            targets = np.argmax(targets, axis=1)

        correct_confidences = clipped_inputs[range(samples), targets]
        negative_log=-np.log(correct_confidences)
        cross_entropy_loss=np.mean(negative_log)
        return cross_entropy_loss

    def backward(self,prob_values, targets):
        samples = len(prob_values)
        labels=targets

        if len(labels.shape) == 2:
            labels = np.argmax(labels, axis=1)

        dvalues = prob_values.copy()
        dvalues[range(samples), labels] -= 1
        return dvalues / samples