
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.res = torch.nn.Sequential(torch.nn.Conv2d(3, 64, 3, 1, 1), torch.nn.Conv2d(64, 64, 1, 1, 0))
        self.shortcut = torch.nn.Sequential(torch.nn.Conv2d(3, 64, 1, 1, 0), torch.nn.PixelShuffle(2))
        self.add = torch.nn.Sequential(torch.nn.ReLU(), torch.nn.Conv2d(64, 64, 1, 1, 0))
    def forward(self, v1):
        split_tensors = torch.split(v1, [1, 1, 1], dim=1)
        concatenated_tensor = torch.cat(split_tensors, dim=1)
        return (concatenated_tensor, torch.split(v1, [1, 1, 1], dim=1))
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
