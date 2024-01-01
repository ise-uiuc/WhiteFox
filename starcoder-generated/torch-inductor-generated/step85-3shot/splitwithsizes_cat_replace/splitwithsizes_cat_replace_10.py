
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.features = torch.nn.ModuleDict({'1': torch.nn.Linear(128, 33), '2': torch.nn.ReLU(inplace=False)})
        self.other_features = torch.nn.BatchNorm2d(128, affine=False)
        self.another_features = torch.nn.Sequential(torch.nn.Linear(33, 128), torch.nn.Linear(128, 33))
    def forward(self, v1):
        split_tensors = torch.split(v1, [1, 1, 1], dim=1)
        out0 = torch.cat(split_tensors, dim=1)
        out1 = self.features[str('1')] (out0)
        out2 = self.features[str('2')] (out1)
        out3 = self.other_features (out2)
        out4 = self.another_features (out2)
        out5 = torch.split(out3, [1, 1, 1], dim=1)
        out6 = torch.cat(out5, dim=1)
        out7 = torch.split(out4, [1, 1, 1], dim=1)
        out8 = torch.cat(out7, dim=1)
        out = [out6, out8]
        return out
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
