
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(in_features=10, out_features=8)
 
    def forward(self, x, other):
        v1 = self.fc(x)
        v2 = v1 + other
        v3 = F.relu(v2)
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(1, 10)
other= torch.randn(1, 8)
