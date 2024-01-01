
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.features = torch.nn.ModuleList([
            torch.nn.BatchNorm1d(32), torch.nn.Sigmoid(), torch.nn.Conv2d(3, 32, 3, 1, 1, bias=True),
            torch.nn.BatchNorm2d(32), torch.nn.LeakyReLU(), torch.nn.Linear(1, 1), torch.nn.Sigmoid(),
            torch.nn.BatchNorm1d(32), torch.nn.Conv2d(32, 32, 3, 1, 1, bias=True),
            torch.nn.BatchNorm2d(32), torch.nn.Conv2d(32, 32, 3, 1, 1, bias=True),
            torch.nn.InstanceNorm2d(32)
        ])
        self.pool = torch.nn.AvgPool2d(2)
    def forward(self, v1):
        split_tensors = torch.split(v1, [1, 1, 1], dim=1)
        concatenated_tensor = torch.cat(split_tensors, dim=1)
        return (concatenated_tensor, torch.split(v1, [1, 1, 1], dim=1))
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
