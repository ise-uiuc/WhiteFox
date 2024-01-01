
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3 * 3 * 12, 1)
 
    def forward(self, x1):
        v1 = self.linear(x1.view(x1.size()[0], -1))
        v2 = v1 - self.other
        v3 = torch.relu(v2)
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3, 32, 32)
