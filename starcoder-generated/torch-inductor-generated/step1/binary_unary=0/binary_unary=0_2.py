
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(6, 16, 3, stride=1, padding=1)
 
    def forward(self, x, other):
        v1 = self.conv(x)
        v2 = v1 + other
        v3 = torch.nn.functional.relu(v2)
        return v3

# Initializing the model
m = Model()
# Inputs to the model, "other"
x = torch.randn(1, 6, 64, 64)
other = torch.randn(1, 16, 32, 32)
