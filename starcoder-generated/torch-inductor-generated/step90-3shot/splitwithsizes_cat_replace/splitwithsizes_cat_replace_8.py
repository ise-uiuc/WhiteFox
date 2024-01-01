
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.features = torch.nn.Sequential(torch.nn.BatchNorm2d(3, affine=True, track_running_stats=True))
    def forward(self, v1):
        split_tensors = torch.split(v1, [1, 1, 1], dim=1)
        concatenated_tensor = torch.cat(split_tensors, dim=1)
        return torch.split(concatenated_tensor, [1, 1, 1], dim=1)
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
