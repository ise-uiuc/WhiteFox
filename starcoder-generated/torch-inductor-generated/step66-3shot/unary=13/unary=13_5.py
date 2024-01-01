
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear0 = torch.nn.Linear(1, 8)
        self.linear1 = torch.nn.Linear(8, 8)
 
    def forward(self, x1):
        v1 = self.linear0(x1)
        v2 = torch.sigmoid(v1)
        v3 = v1 * v2
        v4 = self.linear1(v3)
        return v4

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(2, 1)
