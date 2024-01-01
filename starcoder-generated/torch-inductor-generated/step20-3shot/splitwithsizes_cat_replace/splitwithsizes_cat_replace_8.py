
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.features = torch.nn.Sequential(torch.nn.Conv2d(3, 64, 3, 1, 1, groups=2), torch.nn.Conv2d(64, 64, 3, 2, 3, groups=2), torch.nn.AvgPool2d(3, stride=2, padding=1))
        self.concat = torch.nn.Sequential(torch.nn.Conv2d(64, 64, 3, 1, 1, groups=2))
    def forward(self, v1):
        split_tensors = torch.split(v1, [1, 1, 1], dim=1)
        concatenated_tensor = torch.cat(split_tensors, dim=1)
        return (concatenated_tensor, torch.split(v1, [1, 1, 1], dim=1))
# Inputs to the model
x1 = torch.randn(1, 3, 16, 16)
