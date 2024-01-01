
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.features1 = torch.nn.Sequential(torch.nn.Conv2d(3, 32, 3, 1, 1), torch.nn.MaxPool2d(3, 1, 1, 0))
        self.features2 = torch.nn.Sequential(torch.nn.Conv2d(3, 32, 3, 1, 1))
        self.features = torch.nn.Sequential(self.features1, self.features2)
        self.fc1 = torch.nn.Linear(32, 32)
        self.fc2 = torch.nn.Linear(32, 32)
        self.fc3 = torch.nn.Linear(32, 3)
    def forward(self, x1):
        split_tensors = torch.split(self.features(x1), [1, 1, 1], dim=1)
        concatenated_tensor = torch.cat(split_tensors, dim=1)
        v1 = self.fc1(concatenated_tensor)
        v2 = self.fc2(v1)
        output = self.fc3(v2)
        return (concatenated_tensor, output)
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
