
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.l = torch.nn.Linear(16, 16)
 
    def forward(self, x1, **kwargs):
        v1 = self.l(x1)
        v2 = v1 + kwargs["other"]
        v3 = torch.nn.functional.relu(v2)
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 16)
other = torch.randn(1, 16)
