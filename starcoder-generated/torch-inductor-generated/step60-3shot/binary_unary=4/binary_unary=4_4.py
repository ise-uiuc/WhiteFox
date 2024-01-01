
class Model(torch.nn.Module):
    def __init__(self, other):
        super().__init__()
        self.conv = torch.nn.Linear(64, 128)
        self.other = other
 
    def forward(self, x):
        v1 = self.conv(x)
        v2 = v1 + self.other
        v3 = F.relu(v2)
        return v3
 
# Initializing the model
m = Model(other=torch.randn(1, 64, 1, 1))

# Inputs to the model
x = torch.randn(1, 64, 1, 1)
