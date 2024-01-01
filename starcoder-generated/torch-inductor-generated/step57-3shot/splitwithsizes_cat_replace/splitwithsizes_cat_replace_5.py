
class Model1(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = torch.nn.ReLU()
        self.conv1 = torch.nn.Conv2d(3, 32, 3, 1, 1, bias=False)
        self.conv2 = torch.nn.Conv2d(3, 32, 3, 1, 1, bias=True)
        self.conv3 = torch.nn.Conv2d(3, 32, 3, 1, 1, bias=False)
    def forward(self, v1):
        split_tensors = torch.split(v1, [1, 1, 1], dim=1)
        concatenated_tensor = torch.cat(split_tensors, dim=1)
        conv3_0=self.relu(self.conv3(concatenated_tensor))
        conv1_0=self.relu(self.conv1(concatenated_tensor))
        return (concatenated_tensor, torch.split(conv1_0, [1, 1, 1], dim=1), torch.split(conv3_0, [1, 1, 1], dim=1))
class Model2(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.ModuleList([Model1()])
    def forward(self, v1):
        split_tensors = torch.split(v1, [1, 1, 1], dim=1)
        concatenated_tensor = torch.cat(split_tensors, dim=1)
        return (concatenated_tensor, torch.split(v1, [1, 1, 1], dim=1))
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
