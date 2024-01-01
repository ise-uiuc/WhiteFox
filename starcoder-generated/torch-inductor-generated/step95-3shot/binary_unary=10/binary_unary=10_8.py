
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(in_features=1, out_features=4, bias=True)
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = v1 + 1.1
        v3 = torch.relu(v2)
        return v3

# Initializing the model
m = Model()

# Input to the model
x1 = torch.randn(1, 1)
