
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv3d(1, 32, 1, stride=1, padding=0)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.conv(x)
        x2 = torch.tanh(x1)
        return x2
x = torch.randn(1, 1, 128, 128, 128)
