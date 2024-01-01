
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.features = torch.nn.Sequential(torch.nn.Conv2d(3, 32, 3, 1, 0, bias=False), torch.nn.Conv2d(32, 64, 3, 1, 0))
        self.classifier = torch.nn.Linear(64, 3, bias=False)
    def forward1(self, v1):
        split_tensors = torch.split(v1, [1, 1, 1], dim=1)
        concatenated_tensor = torch.cat(split_tensors, dim=1)
        return (concatenated_tensor, torch.split(v1, [1, 1, 1], dim=1))
class Model1(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.features = torch.nn.Sequential(torch.nn.Conv2d(3, 32, 3, 1, 0, bias=False), torch.nn.Conv2d(32, 64, 3, 1, 0))
    def forward1(self, v1):
        split_tensors = []
        for x1 in torch.split(v1, [1, 1, 1], dim=1):
            split_tensors.append(x1)
        concatenated_tensor = torch.cat(split_tensors, dim=1)
        return (concatenated_tensor, torch.split(v1, [1, 1, 1], dim=1))
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
