
class Model1(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.features = torch.nn.Conv2d(3, 32, 3, 1, 1, bias=False)
    def forward(self, v1):
        split_tensors = torch.split(v1, [1, 1, 1], dim=1)
        concatenated_tensor = torch.cat(split_tensors, dim=1)
        x0 = self.features(concatenated_tensor)
        return (x0, torch.split(v1, [1, 1, 1], dim=1))
class Model2(torch.nn.Module):
    def __init__(self):
        super().__init__()
        block = [Model1()]
        self.features = torch.nn.Sequential(*block * 3)
    def forward(self, v1):
        split_tensors = torch.split(v1, [1, 1, 1, 1], dim=1)
        out = torch.cat(split_tensors, dim=1)[None, :]
        if out.size(1) == 16:
          out = out.permute(1, 0, 2, 3)
# Inputs to the model
x1 = torch.randn(1, 8, 64, 64)
