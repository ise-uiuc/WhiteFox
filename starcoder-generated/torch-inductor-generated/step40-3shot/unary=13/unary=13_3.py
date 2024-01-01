
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(4, 8)
        self.linear2 = torch.nn.Linear(8, 1)
 
    def forward(self, x1):
        v1 = F.relu(self.linear1(x1))
        v2 = torch.sigmoid(self.linear2(v1))
        v3 = v1 * v2
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 4)
