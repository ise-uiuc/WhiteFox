
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(256, 512)
 
    def forward(self, x1):
        v2 = x1 - 11.0
        v2 = v2.repeat((512, 1)).T.repeat((1, 512))
        v3 = self.linear(v2)
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 256)
