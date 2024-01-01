
class Model1(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.features = torch.nn.Sequential(torch.nn.Conv2d(6400, 64, 11, stride=[1, 1], padding=[5, 5], dilation=[2, 2], groups=1, bias=False), torch.nn.ReLU(), torch.nn.GroupNorm(64, 64, eps=1e-05, affine=True, track_running_stats=True), torch.nn.MaxPool2d(kernel_size=2, stride=[2, 2], padding=[0, 0], dilation=[1, 1], ceil_mode=False))
    def forward(self, v1):
        split_tensors = torch.split(v1, [1, 1, 1], dim=1)
        concatenated_tensor = torch.cat(split_tensors, dim=1)
        return (concatenated_tensor, torch.split(v1, [1, 1, 1], dim=1))
class Model2(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.features = torch.nn.Sequential(torch.nn.Conv2d(64, 100, 1, stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=False), torch.nn.ReLU(), torch.nn.GroupNorm(100, 100, eps=1e-05, affine=True, track_running_stats=True), torch.nn.Conv2d(100, 100, 3, stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=False), torch.nn.ReLU(), torch.nn.GroupNorm(100, 100, eps=1e-05, affine=True, track_running_stats=True), torch.nn.Conv2d(100, 32, 1, stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=False), torch.nn.ReLU(), torch.nn.GroupNorm(32, 32, eps=1e-05, affine=True, track_running_stats=True), torch.nn.MaxPool2d(kernel_size=2, stride=[2, 2], padding=[0, 0], dilation=[1, 1], ceil_mode=False), torch.nn.Conv2d(32, 32, 1, stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=False), torch.nn.ReLU(), torch.nn.GroupNorm(32, 32, eps=1e-05, affine=True, track_running_stats=True), torch.nn.Conv2d(32, 32, 3, stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=False), torch.nn.ReLU(), torch.nn.GroupNorm(32, 32, eps=1e-05, affine=True, track_running_stats=True), torch.nn.Conv2d(32, 32, 1, stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=False), torch.nn.ReLU(), torch.nn.GroupNorm(32, 32, eps=1e-05, affine=True, track_running_stats=True), torch.nn.Conv2d(32, 32, 3, stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=False), torch.nn.ReLU(), torch.nn.GroupNorm(32, 32, eps=1e-05, affine=True, track_running_stats=True), torch.nn.ReLU())
    def forward(self, v1):
        split_tensors = torch.split(v1, [1, 1, 1], dim=1)
        concatenated_tensor = torch.cat(split_tensors, dim=1)
        return (concatenated_tensor, torch.split(v1, [1, 1, 1], dim=1))
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
