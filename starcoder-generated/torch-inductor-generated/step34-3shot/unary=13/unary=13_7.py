
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Linear(3, 8)
 
    def forward(self, x1):
        v1 = self.conv(x1)
        s1 = nn.Sigmoid()
        v3 = v1 * s1(v1)
        return v3


# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
