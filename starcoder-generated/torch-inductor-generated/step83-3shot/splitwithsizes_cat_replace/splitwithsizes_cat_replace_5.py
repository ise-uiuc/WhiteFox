
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = torch.nn.ModuleList((torch.nn.Conv2d(8, 8, 1) for _ in torch.arange(1, 11)))
        self.layer2 = torch.nn.ModuleList((torch.nn.BatchNorm2d(8, 1.0, 0.0, True) for _ in torch.arange(1, 11)))
        self.layer3 = torch.nn.ModuleList((torch.nn.BatchNorm2d(8, 1.0, 0.0, False) for _ in torch.arange(1, 11)))
        self.layer4 = torch.nn.ModuleList((torch.nn.Conv2d(8, 8, 3) for _ in torch.arange(1, 11)))
    def forward(self, x):
        split_tensors = torch.split(v1, [1, 1, 1], dim=1)
        concatenated_tensor = torch.cat(split_tensors, dim=1)
        return (concatenated_tensor, torch.split(v1, [1, 1, 1], dim=1))
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
