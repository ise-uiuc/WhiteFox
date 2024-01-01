
class Model(torch.nn.Module):
    def __init__(self, negative_slope):
        super().__init__()
        self.fc = torch.nn.Linear(6, 8)
 
    def forward(self, x):
        v1 = torch.sigmoid(self.fc(x))
        v2 = v1 > 0
        v3 = torch.where(v2, v1, v1 * negative_slope)
        return v3

# Initializing the model
m = Model(0.5)

# Inputs to the model
x = torch.randn(1, 6)
