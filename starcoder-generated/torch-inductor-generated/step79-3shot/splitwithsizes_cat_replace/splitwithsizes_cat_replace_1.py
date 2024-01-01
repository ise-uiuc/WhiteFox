
class Model1(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.block = torch.nn.modules.instancenorm.InstanceNorm2d(32, eps=9.999999747378752e-06, momentum=0.0, affine=False, track_running_stats=True)
    def forward(self, v1):
        split_tensors = torch.split(v1, [1, 1, 1], dim=1)
        concatenated_tensor = torch.cat(split_tensors, dim=1)
        return (split_tensors, concatenated_tensor)
class Model2(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.ModuleList([Model1(), Model1()])
    def forward(self, v1):
        split_tensors = torch.split(v1, [1, 1, 1], dim=1)
        concatenated_tensor = torch.cat(split_tensors, dim=1)
        return (split_tensors, concatenated_tensor)
class Model3(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.features = torch.nn.ModuleList([Model2(), Model2()])
        self.conv = torch.nn.Conv2d(64, 3, 1, 1, 0, bias=False)
    def forward(self, v1):
        split_tensors = torch.split(v1, [1, 1, 1], dim=1)
        concatenated_tensor = torch.cat(split_tensors, dim=1)
        model = self.features[1]
        split_tensors = model(concatenated_tensor)
        concatenated_tensor = torch.cat(split_tensors, dim=1)
        return (concatenated_tensor, torch.split(v1, [1, 1, 1], dim=1))
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
