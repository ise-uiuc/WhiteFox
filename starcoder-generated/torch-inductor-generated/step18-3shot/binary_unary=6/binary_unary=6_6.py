
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(288, 784)
 
    def forward(self, y):
        h1 = self.linear(y)
        h2 = h1 - 1
        h3 = torch.relu(h2)
        return h3

# Initializing the model
m = Model()

# Inputs to the model
y = torch.randn(1, 288)
