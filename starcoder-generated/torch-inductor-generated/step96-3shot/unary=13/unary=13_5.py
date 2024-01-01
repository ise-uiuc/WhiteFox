
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(16, 6)
 
    def forward(self, x67):
        v66 = self.linear(x67)
        v68 = torch.sigmoid(v66)
        v69 = v66 * v68
        return v69

# Initializing the model
m = Model()

# Inputs to the model
x67 = torch.randn(1, 16)
