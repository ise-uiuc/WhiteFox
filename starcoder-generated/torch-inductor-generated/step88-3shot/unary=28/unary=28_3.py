
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(10, 5)
 
    def forward(self, x1, **kws):
        v1 = self.fc(x1)
        v2 = torch.clamp(v1, **kws)
        return v2

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(4, 5)
