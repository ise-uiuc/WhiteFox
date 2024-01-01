
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 10)
 
    def forward(self, x1):
        x2 = self.linear(x1)
        x3 = x2 - 5
        x4 = F.relu(x3)
        return x4

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.ones(1, 10)
