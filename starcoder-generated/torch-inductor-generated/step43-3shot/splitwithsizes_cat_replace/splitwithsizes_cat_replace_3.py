
class Model1(torch.nn.Module):
    def __init__(self):
        super().__init__()
        block_0 = [torch.nn.BatchNorm2d(16, affine=False, track_running_stats=False)]
        block_1 = [torch.nn.BatchNorm2d(16, affine=False, track_running_stats=False)]
        block_2 = [torch.nn.ReLU()]
        block_3 = [torch.nn.ReLU()]
        block_4_left = [torch.nn.MaxPool2d(kernel_size=[3, 3], stride=[2, 2], padding=[1, 1])]
        block_4_right = [torch.nn.Conv2d(16, 16, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1]), torch.nn.BatchNorm2d(16, affine=False, track_running_stats=False)]
        self.features = torch.nn.Sequential(*block_0, *block_1, *block_2, *block_3, *block_4_left, *block_4_right)
    def forward(self, v1):
        split_tensors = torch.split(v1, [1, 1, 1], dim=1)
        concatenated_tensor = torch.cat(split_tensors, dim=1)
        return (concatenated_tensor, torch.split(v1, [1, 1, 1], dim=1))
class Model2(torch.nn.Module):
    def __init__(self):
        super().__init__()
        block_0 = [torch.nn.BatchNorm2d(16, affine=False, track_running_stats=False)]
        block_1 = [torch.nn.BatchNorm2d(16, affine=False, track_running_stats=False)]
        block_2 = [torch.nn.ReLU()]
        block_3 = [torch.nn.ReLU()]
        block_4_left = [torch.nn.MaxPool2d(kernel_size=[3, 3], stride=[2, 2], padding=[1, 1])]
        block_4_right = [torch.nn.Conv2d(16, 16, kernel_size=[3, 3], stride=[2, 2], padding=[1, 1]), torch.nn.BatchNorm2d(16, affine=False, track_running_stats=False)]
        self.features = torch.nn.Sequential(*block_0, *block_1, *block_2, *block_3, *block_4_left, *block_4_right)
    def forward(self, v1):
        split_tensors = torch.split(v1, [1, 1, 1], dim=1)
        concatenated_tensor = torch.cat(split_tensors, dim=1)
        return (concatenated_tensor, torch.split(v1, [1, 1, 1], dim=1))
# Inputs to the model
x1 = torch.randn(1, 16, 112, 112)
x2 = torch.randn(1, 16, 112, 112)
x3 = torch.randn(1, 16, 56, 56)
