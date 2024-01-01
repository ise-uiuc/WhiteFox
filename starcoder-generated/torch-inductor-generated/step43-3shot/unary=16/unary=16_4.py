
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 5)
 
    def forward(self, x1):
        y1 = torch.relu(self.linear(x1))
        return y1

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3)
