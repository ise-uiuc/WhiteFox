
# PyTorch version: 1.9.0  
alpha = 1.6732632423543772848170429916717
scale = 1.0507009873554804934193349852946
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(8, 8)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = v1 * torch.clamp(torch.minimum(torch.maximum(v1 + 3, 0), 6))
        v3 = v2 / 6
        return v3 * scale

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 8)
