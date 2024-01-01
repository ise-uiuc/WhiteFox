
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(3, 4, bias=True)
        self.linear2 = torch.nn.Linear(4, 8, bias=True)
 
    def forward(self, x1):
        v1 = self.linear1(x1)
        v2 = v1 - 1
        v3 = F.relu(v2)
        v4 = self.linear2(v3)
        return v4

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3)

