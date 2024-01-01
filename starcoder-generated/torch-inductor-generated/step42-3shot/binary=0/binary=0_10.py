
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
    def forward(self):
        v1 = torch.randn(1, 8, 64, 64)
        v2 = torch.nn.functional.relu(v1)
        return v2
# Inputs to the model
