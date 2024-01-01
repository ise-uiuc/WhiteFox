
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.features1 = torch.nn.Sequential(torch.nn.Conv2d(3, 32, 3, 1, 1, bias=False), torch.nn.BatchNorm2d(32, affine=False, track_running_stats=False)),
        self.features2 = torch.nn.Sequential(torch.nn.Conv2d(32, 32, 3, 1, 1, bias=False), torch.nn.BatchNorm2d(32, affine=False, track_running_stats=False)),
        self.features3 = torch.nn.Sequential(torch.nn.Conv2d(32, 32, 3, 1, 1, bias=False), torch.nn.BatchNorm2d(32, affine=False, track_running_stats=False)),
    def forward(self, v1, v2, v3):
        split_tensors1 = torch.split(v1, [1, 1, 1], dim=1)
        concatenated_tensor1 = torch.cat(split_tensors1, dim=1)
        split_tensors2 = torch.split(v2, [1, 1], dim=1)
        concatenated_tensor2 = torch.cat(split_tensors2, dim=1)
        split_tensors3 = torch.split(v3, [1, 1, 1], dim=1)
        concatenated_tensor3 = torch.cat(split_tensors3, dim=1)
        return (concatenated_tensor1, concatenated_tensor2, concatenated_tensor3)
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
x2 = torch.randn(1, 6, 64, 50)
x3 = torch.randn(1, 9, 64, 50)
