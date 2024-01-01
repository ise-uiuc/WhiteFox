
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.features = torch.nn.Sequential(torch.nn.MaxPool2d(3, 2, 0, 1), torch.nn.MaxPool2d(2, 1, (0,), 0), torch.nn.MaxPool2d(4, (1,), 2, 1, False, False, (2, 3), (1, 2), 2, False, True))
        self.split = torch.nn.Sequential(torch.nn.Conv2d(3, 3, 3, 1, 1), torch.nn.Conv2d(3, 3, 3, 2, 3), torch.nn.Conv2d(3, 3, 5, 4, 1))
    def forward(self, v1):
        split_tensors = torch.split(v1, [1, 1, 1], dim=1)
        concatenated_tensor = torch.cat(split_tensors, dim=1)
        return (concatenated_tensor, torch.split(v1, [1, 1, 1], dim=1))
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
