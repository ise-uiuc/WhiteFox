
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.features1 = torch.nn.Conv2d(128, 32, 1, 1, 0)
        self.features2 = torch.nn.MaxPool2d(2, 1, 1)
        self.features3 = torch.nn.MaxPool2d(3, 1, 1)
        self.features4 = torch.nn.BatchNorm2d(32, affine=False)
    def forward(self, v1):
        split_tensors = torch.split(v1, [1, 1, 1], dim=1)
        concatenated_tensor = torch.cat(split_tensors, dim=1)
        return (concatenated_tensor, torch.split(v1, [1, 1, 1], dim=1))
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
