
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Sequential(
            torch.nn.Linear(16, 32),
        )
        self.other = torch.rand(1, 32)
 
    def forward(self, X1):
        v1 = self.linear(X1)
        v2 = v1 + self.other
        v3 = F.relu(v2)
        return v3
 
# Initializing the model
m = Model()

# Inputs to the model
X1 = torch.randn(4, 16)
