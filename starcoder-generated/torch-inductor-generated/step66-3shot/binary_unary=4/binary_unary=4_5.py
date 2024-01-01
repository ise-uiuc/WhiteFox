
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(8, 3)
 
    def forward(self, x1, **kwargs):
        other = kwargs.get("other")
        v1 = self.linear(x1)
        v2 = v1 + other
        v3 = torch.relu(v2)
        return v3
 
# Initializing the model
other = torch.randn(1, 3)
m = Model()

# Inputs to the model
x1 = torch.randn(1, 8)
