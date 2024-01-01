
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(100, 200)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        a1 = torch.relu(v1)
        return a1

# Inputs to the model
x1 = torch.randn(1, 100)
