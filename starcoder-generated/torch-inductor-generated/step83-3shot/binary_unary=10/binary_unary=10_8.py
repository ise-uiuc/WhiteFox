
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(8, 8)
        self.linear2 = torch.nn.Linear(8, 8)
 
    def forward(self, x1):
        v1 = self.linear1(x1)
        v2 = v1 + torch.randn(8)
        v3 = torch.nn.functional.relu(v2)
        v4 = self.linear2(v3)
        return v4

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 8)
