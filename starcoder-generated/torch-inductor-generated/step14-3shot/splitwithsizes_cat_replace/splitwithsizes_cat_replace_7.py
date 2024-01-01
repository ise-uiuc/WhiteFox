
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.features = torch.nn.Sequential(torch.nn.Conv2d(3, 3, 3, 1, 1))
        self.split = torch.nn.Sequential(torch.nn.MaxPool2d(6, 2, 3, 1), torch.nn.MaxPool2d(2, 3, 1, 2))
    def forward(self, x1):
        v1 = self.features(x1)
        split_tensors = torch.split(v1, [6, 100, 133], dim=2)
        concatenated_tensor = torch.cat(split_tensors, dim=2)
        return (concatenated_tensor, torch.split(v1, [6, 100, 133], dim=2))
# Inputs to the model
x1 = torch.randn(1, 3, 128, 128)
