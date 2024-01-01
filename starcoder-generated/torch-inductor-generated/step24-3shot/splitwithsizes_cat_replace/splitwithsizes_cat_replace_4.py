
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.features = torch.nn.Sequential(torch.nn.Conv2d(3, 32, 3, 1, 1), torch.nn.ReLU(inplace=False), torch.nn.Conv2d(32, 32, 3, 1, 1), torch.nn.ReLU(inplace=False), torch.nn.Conv2d(32, 32, 3, 2, 3), torch.nn.ReLU(inplace=False), torch.nn.Conv2d(32, 32, 3, 1, 1), torch.nn.ReLU(inplace=False))
        self.concat = torch.nn.Sequential(torch.nn.Conv2d(32, 32, 3, 1, 0))
        self.softmax = torch.nn.Softmax(dim=0)
    def forward(self, v1):
        split_tensors = torch.split(v1, [1, 1, 1], dim=1)
        concatenated_tensor = torch.cat(split_tensors, dim=1)
        return self.softmax(torch.relu(concatenated_tensor))
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
