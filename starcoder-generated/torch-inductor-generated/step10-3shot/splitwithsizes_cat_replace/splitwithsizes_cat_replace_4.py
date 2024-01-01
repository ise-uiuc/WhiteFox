
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.features = torch.nn.Sequential(torch.nn.MaxPool2d(12, 8, 6, 1), torch.nn.MaxPool2d(10, 8, 5, 3), torch.nn.MaxPool2d(6, 7, 4, 2))
        self.split = torch.nn.Sequential(torch.nn.MaxPool2d(6, 5, 3, 1), torch.nn.MaxPool2d(3, 4, 2, 0), torch.nn.MaxPool2d(6, 2, 3, 1), torch.nn.MaxPool2d(3, 1, 2, 0), torch.nn.MaxPool2d(7, 7, 4, 0), torch.nn.MaxPool2d(6, 3, 5, 3), torch.nn.MaxPool2d(12, 4, 7, 0), torch.nn.MaxPool2d(4, 1, 4, 1), torch.nn.MaxPool2d(5, 5, 7, 2), torch.nn.MaxPool2d(6, 2, 6, 2), torch.nn.MaxPool2d(2, 2, 2, 1))
        self.flatten = torch.nn.Flatten()
    def forward(self, x1):
        v1 = self.features(x1)
        v2 = self.features(x1)
        concat_tensor = torch.cat([v1, v2], dim=-1)
        split_tensors = torch.split(concat_tensor, [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dim=1)
        concatenated_tensor = torch.cat(split_tensors, dim=1)
        return self.flatten(concatenated_tensor)
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
