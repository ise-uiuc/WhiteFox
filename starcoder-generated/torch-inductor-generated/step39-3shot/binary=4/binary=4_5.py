
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear45 = torch.nn.Linear(133, 93)
 
    def forward(self, v1, t1):
        v2 = self.linear45(v1)
        v3 = torch.add(v2, t1, alpha=1.0)
        return v3

# Initializing the model
m = Model()

# Inputs to the model
v1 = torch.randn(2, 133)
t1 = torch.randn(2, 93)
