
class Model(torch.nn.Module):
    def __init__(self, other):
        super().__init__()
        self.linear = torch.nn.Linear(10, 10)
 
    def forward(self, x1, **kwargs):
        v1 = self.linear(x1)
        v2 = v1 + kwargs["other"]
        v3 = torch.nn.functional.relu(v2)
        return v3

# Initializing the model
m = Model(torch.randn(1, 10))

# Inputs to the model
x1 = torch.randn(1, 10)
