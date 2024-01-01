
class Model(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = torch.nn.Conv1d(in_channels=15, out_channels=3, kernel_size=3, padding=1)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = torch.max(v1, dim=1)[0]
        v3 = torch.sigmoid(v2)
        return v3
# Inputs to the model
x1 = torch.randn(1, 15, 40)
