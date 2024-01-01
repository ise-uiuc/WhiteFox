
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(16, 8)
 
    def forward(self, x):
        v1 = self.linear(x)
        v2 = v1 + 1 # Initialize "other"
        return v2

y1 = torch.randn(1, 6, 8, 8)
___output___ = m(y1)

# Inputs to the model
