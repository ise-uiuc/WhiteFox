
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(8, 8, False)
 
    def forward(self, x1):
        t1 = self.conv(x1)
        v2 = v1 - 1
        v3 = F.relu(v2)
        v4 = F.sigmoid(v3)
        return v5

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 8)
x2 = torch.randn(1, 8)
