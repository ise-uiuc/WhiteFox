
class Model1(torch.nn.Module):
    def __init__(self):
        super().__init__()
        block_0 = [torch.nn.Conv2d(3, 32, 2, 2, 0, bias=False)]
        block_1_left = [torch.nn.ReLU()]
        block_1_right = [torch.nn.Conv2d(32, 64, 2, 1, 0, bias=False), torch.nn.BatchNorm2d(64, affine=False, track_running_stats=False)]
        self.features = torch.nn.Sequential(*block_0, *block_1_left, *block_1_right)
    def forward(self, v1):
        split_tensors = torch.split(v1, [1, 1, 1], dim=1)
        concatenated_tensor = torch.cat(split_tensors, dim=1)
        concatenated_tensors = torch.cat(split_tensors, dim=0)
        return (concatenated_tensor, torch.split(v1, [1, 1, 1], dim=1))
class Model2(torch.nn.Module):
    def __init__(self):
        super().__init__()
        block_left = [torch.nn.Linear(3, 32), torch.nn.ReLU()]
        block_right = [torch.nn.Linear(96, 3)]
        self.features = torch.nn.Sequential(*block_left, *block_right)
    def forward(self, v1):
        split_tensors = torch.split(v1, [1, 1, 1], dim=1)
        concatenated_tensor = torch.cat(split_tensors, dim=1)
        concatenated_tensor = torch.cat([v1, v1])
        return (concatenated_tensor, torch.split(concatenated_tensor, [1, 1, 1], dim=1))
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
