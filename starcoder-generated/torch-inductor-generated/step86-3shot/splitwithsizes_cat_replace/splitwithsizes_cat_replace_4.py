
class Model1(torch.nn.Module):
    def __init__(self):
        super().__init__()
        block = [torch.nn.Conv2d(3, 32, 1, 1, 0, bias=False)]
        self.features = torch.nn.Sequential(*block * 8)
    def forward(self, v):
        split_tensors = torch.split(v, [1, 1, 1], dim=1)
        return (torch.cat(split_tensors, dim=1), split_tensors)
class Model2(torch.nn.Module):
    def __init__(self):
        super().__init__()
        moduleList = []
        for i in range(6):
            moduleList.extend([torch.nn.Conv2d(32, 32, 3, 1, 1, bias=False), torch.nn.BatchNorm2d(32)])
        self.moduleList = torch.nn.Sequential(*moduleList)
        self.features = torch.nn.Sequential(Model1(), self.moduleList, Model1(), torch.nn.Flatten(), torch.nn.Linear(32, 3))
    def forward(self, v):
        return self.features(torch.cat([v] * 2))
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
