
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(256, 128, bias=False)
 
    def forward(self, x1, other):
        v1 = self.linear(x1)
        v2 = v1 + other 
        v3 = torch.relu(v2)
        return v3

# Inputs to the model
x1 = torch.randn(1, 256)
x2 = torch.randn(1, 128)
