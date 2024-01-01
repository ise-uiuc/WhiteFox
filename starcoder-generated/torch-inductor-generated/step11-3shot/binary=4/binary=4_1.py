
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(128, 64)
 
    def forward(self, x1):
        return self.linear(x1) + x1

# Inputs to the model
x1 = torch.randn(64, 128)
