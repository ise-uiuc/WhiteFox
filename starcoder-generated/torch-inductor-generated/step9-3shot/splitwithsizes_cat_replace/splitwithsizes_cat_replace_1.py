
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.features = torch.nn.Sequential(torch.nn.Conv2d(3, 32, 3, 1, 1), torch.nn.Conv2d(32, 3, 3, 1, 1))
        self.split = torch.nn.Sequential(torch.nn.Conv2d(3, 3, 3, 2, 1, groups=3), torch.nn.Conv2d(3, 3, 5, 4, 2, groups=3), torch.nn.Conv2d(3, 3, 3, 1, 0, groups=3))
    def forward(self, x1):
        v1 = self.features(x1)
        split_tensors = self.split(v1)
        concatenated_tensor = torch.cat(split_tensors, dim=1)
        return (concatenated_tensor, split_tensors)
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
