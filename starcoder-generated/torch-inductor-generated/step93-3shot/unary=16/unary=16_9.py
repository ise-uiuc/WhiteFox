
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(157, 92)
 
    def forward(self, x1_1):
        v1 = self.linear(x1_1)
        v2 = torch.relu(v1)
        return v2

# Initializing the model
m = Model()

# Inputs to the model
x1_1 = torch.randn(1, 157)
