
class Model3(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.features = torch.nn.PixelShuffle(2)
    def forward(self, v1):
        split_tensors = torch.split(v1, [4, 4, 4], dim=3)
        return torch.split(self.features(split_tensors), [1, 1, 1], dim=3)
# Inputs to the model
x1 = torch.randn(1, 4, 6, 8)
