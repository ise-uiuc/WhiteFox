
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(728, 2048)
 
    def forward(self, x1):
        v1 = self.fc(x1)
        v2 = v1 > 0
        v3 = torch.where(v2, v1, v1 * negative_slope)
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 728)
