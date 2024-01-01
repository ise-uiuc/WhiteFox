
class MyModel (torch.nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.features = torch.nn.Sequential(torch.nn.Conv2d(3, 32, 3, 1, 1), torch.nn.ReLU(inplace=False), torch.nn.MaxPool2d(3, 2, 1, 1), torch.nn.Conv2d(32, 32, 5, 3, 2), torch.nn.BatchNorm2d(32), torch.nn.ReLU(inplace=False), torch.nn.Conv2d(32, 32, 3, 1, 1), torch.nn.ReLU(inplace=False))
        self.concat = torch.nn.Sequential(torch.nn.Conv2d(64, 32, 3, 2, 1))
    def forward(self, x):
        v1 = self.features(x)
        split_tensors = torch.split(v1, [1, 1, 1], dim=1)
        concatenated_tensor = torch.cat(split_tensors, dim=1)
        return (concatenated_tensor, torch.split(v1, [1, 1, 1], dim=1))
# Inputs to the model
x1 = torch.randn(32, 3, 64, 64)
