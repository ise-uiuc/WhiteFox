
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv1d(3, 16, 1, stride=1, padding=1)
    def forward(self, x):
        v1 = self.conv(x)
        v2 = v1 - torch.tensor([0.754, 0.12123, 0.8424, 0.2234, 0.7123, 0.934], dtype=torch.float32).reshape((6, 1, 1)).to(device)
        return v2
# Inputs to the model
x = torch.randn(1, 3, 32)
