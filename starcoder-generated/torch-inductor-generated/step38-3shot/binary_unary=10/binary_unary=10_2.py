
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.in_features = 4096
        self.out_features = 10
        self.weight = torch.nn.parameter.Parameter(torch.ones(self.in_features, self.out_features))
 
        def forward(self, x1):
            y = torch.matmul(x1, self.weight)
            y1 = y + y 
            return torch.relu(y1)

# Initializing the model
import torch
m = Model(3)

# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
