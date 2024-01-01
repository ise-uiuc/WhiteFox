
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(28 * 28, 10)
 
    def forward(self, x1):
        v1 = x1.view(-1, 28 * 28)
        v2 = self.linear(v1)
        v3 = torch.sigmoid(v2)
        return v2

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 28, 28)
