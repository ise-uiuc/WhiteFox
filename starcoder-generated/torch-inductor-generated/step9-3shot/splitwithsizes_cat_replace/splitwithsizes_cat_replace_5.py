
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.features_1 = torch.nn.Sequential(torch.nn.Conv2d(3, 32, 3, 1, 1), torch.nn.Conv2d(32, 32, 3, 1, 1))
        self.features_2 = torch.nn.Sequential(torch.nn.Conv2d(32, 3, 3, 1, 1))
        self.split_1 = torch.nn.Sequential(torch.nn.Conv2d(3, 32, 3, 1, 1), torch.nn.MaxPool2d(-1, 1, 1, 0))
        self.split_2 = torch.nn.Sequential(torch.nn.MaxPool2d(3, -1, 1, 0), torch.nn.Conv2d(32, 3, 3, 1, 1))
    def forward(self, x1):
        v1 = self.features_1(x1)
        v2 = self.features_2(v1)
        split_tensors = torch.split(v1, [1, 1, 1], dim=1)
        concatenated_tensor = torch.cat(split_tensors, dim=1)
        return [None, v2]
# Inputs to the model
x1 = torch.randn(1, 3, 32, 32)
