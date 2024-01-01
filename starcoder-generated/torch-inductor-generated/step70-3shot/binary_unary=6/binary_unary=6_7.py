
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(8 + 8, 16)
 
    def forward(self, x1):
        v1 = self.linear1(torch.cat((x1, x2), dim=1)
        v2 = v1 - other
        v3 = F.relu(v2)
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x2 = torch.randn(1, 8)
other = torch.rand(1)
