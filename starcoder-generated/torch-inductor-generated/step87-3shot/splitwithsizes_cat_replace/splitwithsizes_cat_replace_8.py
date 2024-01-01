
class Block(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.bn1 = torch.nn.BatchNorm2d(64, affine=False, track_running_stats=False)
        self.bn2 = torch.nn.BatchNorm2d(64, affine=False, track_running_stats=False)
    def forward(self, input1):
        split_tensors = torch.split(input1, [1, 1, 1], dim=1)
        concatenated_tensor = torch.cat(split_tensors, dim=1)
        return (concatenated_tensor, torch.split(input1, [1, 1, 1], dim=1))
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.features = torch.nn.Sequential(Block(), torch.nn.ReLU(), torch.nn.functional.interpolate, torch.nn.BatchNorm1d(8, affine=False, track_running_stats=False))
    def forward(self, v1):
        split_tensors = torch.split(v1, [1, 1, 1], dim=1)
        concatenated_tensor = torch.cat(split_tensors, dim=1)
        return (concatenated_tensor, torch.split(v1, [1, 1, 1], dim=1))
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
