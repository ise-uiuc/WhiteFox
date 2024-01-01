
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(64)
        
    def forward(self, x1, x2):
        v1 = self.linear1(x1)
        v2 = v1 + x2
        return v2

# Initializing the model
p = 1
m = Model()
o = torch.randn(1, 64)
