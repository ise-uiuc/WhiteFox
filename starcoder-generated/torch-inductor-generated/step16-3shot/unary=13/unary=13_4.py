
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(12, 20)
 
    def forward(self, x2):
        v1 = self.linear(x2)
        v2 = torch.sigmoid(v1)
        v3 = v1 * v2
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x2 = torch.randn(1, 12)
