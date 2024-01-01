
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 128, bias=True)
        self.relu = torch.nn.ReLU()
 
    def forward(self, x1, x2):
        v1 = self.linear(x1)
        v2 = v1 - x2
        v3 = self.relu(v2)
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(2, 10)
x2 = torch.randn(1)
