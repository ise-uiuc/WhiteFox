
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 64, 3, 1, 1)
        self.relu1 = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv2d(64, 64, 3, 1, 1)
        self.relu2 = torch.nn.ReLU()
    def forward(self, v1):
        split_tensors = torch.split(v1, [1, 1, 1], dim=1)
        concatenated_tensor = torch.cat(split_tensors, dim=1)
        v2 = self.relu1(self.conv1(concatenated_tensor))
        v3 = torch.split(v1, [1, 1, 1], dim=1)
        concatenated_tensor = torch.cat(v3, dim=1)
        v4 = self.relu2(self.conv2(concatenated_tensor))
        return (concatenated_tensor, torch.split(v1, [1, 1, 1], dim=1))
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
