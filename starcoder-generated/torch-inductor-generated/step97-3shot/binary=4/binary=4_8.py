
import torch
class MLPBlock3(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.W = torch.nn.Linear(input_size, output_size)
        self.add_input = input_size == output_size

    def forward(self, x1):
        v1 = self.W(x1)
        if self.add_input:
            v2 = v1 + x1
        else:
            raise NotImplementedError(
                "If the input size is different from the output size, please implement the line below:")
            v2 = v1 + <TODO: Please implement what you need here>
        return v2

# Initializing and setting parameters of the model
input_size = 5
output_size = 3
n_features = 2
m = MLPBlock3(input_size, output_size)
torch.manual_seed(123)
m.W.weight.data.fill_(0.3)
m.W.bias.data.fill_(0)

# Inputs to the model
x1 = torch.randn(n_features)
