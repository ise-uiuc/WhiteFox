
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(16, 32)
 
    def forward(self, x10, x11):
        v10 = self.linear(x10)
        v11 = self.linear(x11)
        v12 = v10 + v11
        v9 = v12
        v13 = torch.nn.functional.relu(v9)
        return v13

# Initializing the model
m = Model()

# Inputs to the model
x10 = torch.randn(1, 16)
x11 = torch.randn(1, 16)
