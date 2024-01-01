
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(1, 50)
        self.relu = torch.nn.ReLU()
 
    def forward(self, x1, x2):
        v1 = self.linear(x1)
        return self.relu(v1)

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 7)
x2 = torch.randn(1, 3)
