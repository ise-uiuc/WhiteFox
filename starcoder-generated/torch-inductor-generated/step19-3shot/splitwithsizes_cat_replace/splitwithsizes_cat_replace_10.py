
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.features = torch.nn.Sequential(torch.nn.Conv2d(3, 64, 3, 1, 1), torch.nn.MaxPool2d(3, 2, 1, 1))
        self.split_tensors = torch.nn.Sequential(torch.nn.MaxPool2d(3, 1, 1, 0))
        self.concat = torch.nn.Sequential(torch.nn.MaxPool2d(3, 2, 1, 1))
    def forward(self, x1):
        v1 = self.features(x1)
        split_tensors = torch.split(v1, [1, 1, 64], dim=1)
        concatenated_tensor = torch.cat(split_tensors, dim=1)
        return (concatenated_tensor, torch.split(v1, [1, 1, 64], dim=1))
# Inputs to the model
x1 = torch.randn(32, 3, 64, 64)
