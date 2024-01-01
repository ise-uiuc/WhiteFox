
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(16, 1)
        self.linear2 = torch.nn.Linear(16, 1)
 
    def forward(self, x):
        v1 = self.linear1(x)
        v2 = v1 + self.linear2(x)
        v3 = F.relu(v2)
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(1, 16)
