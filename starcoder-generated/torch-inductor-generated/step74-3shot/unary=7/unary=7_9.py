
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(4096, 2048)
 
    def forward(self, x1):
        v1 = self.linear1(x1)
        v2 = v1 * torch.clamp(torch.min(torch.max(v1 + 3, 0), 6),0,float('inf'))
        v3 = v2 / 6
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 4096)
