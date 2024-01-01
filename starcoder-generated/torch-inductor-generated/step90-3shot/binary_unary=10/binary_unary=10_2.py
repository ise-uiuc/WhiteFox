
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(8, 16)
        self.linear2 = torch.nn.Linear(16, 32)
 
    def forward(self, x1):
        v1 = self.linear1(x1)
        v2 = v1 + self.linear2(x1)
        v3 = F.relu(v2)
        return v3


# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 8)
