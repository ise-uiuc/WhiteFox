
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.features = [torch.nn.Conv2d(3, 32, 3, 1, 1, bias=False)]
        self.features_1 = [torch.nn.Conv2d(32, 32, 3, 1, 1, bias=False)]
        self.features_2 = [torch.nn.BatchNorm2d(32)]
        self.features_3 = [torch.nn.ReLU()]
        self.features_4 = [torch.nn.Conv2d(32, 32, 3, 1, 1, bias=False), torch.nn.Conv2d(32, 32, 3, 1, 1, bias=False), torch.nn.ReLU(), torch.nn.Conv2d(32, 32, 3, 1, 1, bias=False), torch.nn.Conv2d(32, 32, 3, 1, 1, bias=False)]
        self.features = torch.nn.Sequential(*self.features, *self.features_1, *self.features_2, *self.features_3, *self.features_4)
        self.extra = torch.nn.ReLU()
    def forward(self, v1):
        split_tensors = torch.split(v1, [1, 1, 1], dim=1)
        concatenated_tensor = torch.cat(split_tensors, dim=1)
        v2, v3, v5, v6 = split_tensors
        return torch.nn.ReLU()(torch.nn.functional.instance_norm(concatenated_tensor))
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
