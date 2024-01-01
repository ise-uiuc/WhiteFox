
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.features = torch.nn.Sequential(torch.nn.Conv2d(3, 32, 3, 1, 3), torch.nn.Conv2d(32, 3, 3, 1, 1))
        self.split = torch.nn.Sequential(torch.nn.MaxPool2d(7, 2, 0, 0))
    def forward(self, x1):
        v1 = self.features(x1)
        split_tensors = torch.split(v1, [1, 1, 1], dim=1)
        concatenated_tensor = torch.cat(split_tensors, dim=1)
        return (concatenated_tensor, torch.split(v1, [1, 1, 1], dim=1))
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
