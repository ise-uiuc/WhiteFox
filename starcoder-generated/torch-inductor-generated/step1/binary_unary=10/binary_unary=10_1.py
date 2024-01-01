
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(16, 8, bias=True)
 
    def forward(self, x):
        v1 = self.linear(x)
        v2 = {"other": 0.3}
        v3 = v1 + v2
        v4 = nn.functional.relu(v3)
        return v4

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(8, 16)
