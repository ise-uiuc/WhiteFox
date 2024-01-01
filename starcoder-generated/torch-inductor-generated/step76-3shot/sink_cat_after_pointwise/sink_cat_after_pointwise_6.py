
import torch
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 12)

    def forward(self, input):
        y = self.linear(input)
        output = y.view(y.size(0), -1) if y.size(1) == 12 else y.Tanh()
        output = output.tanh()  # the only user of y, sink pointwise op before
        return output
# Inputs to the model
x = torch.randn(2, 3)
