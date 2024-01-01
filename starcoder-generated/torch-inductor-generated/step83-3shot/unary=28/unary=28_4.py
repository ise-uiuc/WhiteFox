
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(512, 1024)
 
    def forward(self, x1, y1):
        v1 = self.linear(x1)
        t1 = torch.clamp_min(v1, y1)
        v2 = torch.clamp_max(v1, y1)
        return t1, v2

# Initializing the model
m = Model()

# Inputs of the model
x1 = torch.randn(1, 512)
y1 = 1
