
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.features = torch.nn.ModuleList()
        for i in range(3):
            self.features.append(torch.nn.Conv2d(3, 32, 3, 1, 1, bias=True))
            self.features.append(torch.nn.BatchNorm2d(32))
            self.features.append(torch.nn.ReLU())
            self.features.append(torch.nn.ModuleList())
            for j in range(3):
                self.features[i].append(torch.nn.Linear(1, 1))       
                self.features[i].append(torch.nn.Linear(1, 1))
    def forward(self, v1):
        split_tensors = torch.split(v1, [1, 1, 1], dim=1)
        concatenated_tensor = torch.cat(split_tensors, dim=1)
        return (concatenated_tensor, torch.split(v1, [1, 1, 1], dim=1))
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
