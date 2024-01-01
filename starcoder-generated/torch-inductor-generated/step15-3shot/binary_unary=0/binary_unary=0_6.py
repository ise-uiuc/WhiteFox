
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Linear(128, 128)
    def forward(self, x):
        v1 = self.conv(x)
        v2 = torch.softmax(v1)
        v3 = v2 + x
        return v3
# Inputs to the model
x = torch.randn(1, 128)
