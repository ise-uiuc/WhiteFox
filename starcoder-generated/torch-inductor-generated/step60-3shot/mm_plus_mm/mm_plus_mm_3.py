
class Model(torch.nn.Module):
    def forward(self, weights, inputs):
        return torch.softmax(sum(w * i for w, i in zip(weights, inputs)), -1)
# Inputs to the model
input_A = torch.tensor([0.09, 0.12, 0.23, 0.42, 0.19, 0.37, 0.88, 0.32, 0.76, 0.5, 0.81, 0.67])
input_B = torch.tensor([0.85, 0.81, 0.29, 0.82, 0.76, 0.38, 0.91, 0.16, 0.76, 0.51, 0.51, 0.98])
input_C = torch.tensor([0.79, 0.86, 0.9, 0.21, 0.22, 0.72, 0.24, 0.91, 0.88, 0.54, 0.32, 0.7])
inputs = [input_A, input_B, input_C]
weight_A = torch.tensor([7.58, 2.12, 8.58, 0.48, 1.11, 5.88, 9.32, 6.9, 8.5, 6.41, 2.31, 4.98])
weight_B = torch.tensor([8.36, 9.37, 5.53, 3.57, 3.68, 1.86, 4.26, 6.53, 9.44, 7.47, 9.82, 7.93])
weight_C = torch.tensor([4.68, 9.84, 5.26, 8.48, 7.49, 0.65, 9.05, 1.39, 0.67, 7.76, 3.4, 5.66])
weights = [weight_A, weight_B, weight_C]
