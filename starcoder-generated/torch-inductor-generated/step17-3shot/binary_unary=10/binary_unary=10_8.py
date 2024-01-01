
class Model(torch.nn.Module):
    def __init__(self):
        super(Model1, self).__init__()
        self.linear = torch.nn.Linear(2, 8)
 
    def forward(self, x1, x2):
        v1 = self.linear(x1)
        v2 = x2 + v1
        v3 = torch.relu(v2)
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 2)
x2 = torch.randn(1, 8)
