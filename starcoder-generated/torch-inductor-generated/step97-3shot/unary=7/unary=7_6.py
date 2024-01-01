
import torch.nn.functional as F

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(128, 256)
 
    def forward(self, input):
        y = self.linear(input)
        y_clamp = F.hardtanh(y, 0., 6.)
        y_scaled = y_clamp + 3.
        y2 = y_scaled / 6
        return y2

m = Model()

# Inputs to the model
input = torch.randn(256, 128)
