
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        block = [torch.nn.Conv2d(3, 32, 3, 1, 1, bias=False)]
        self.features = torch.nn.Sequential(*block * 3)
    def forward(self, v1):
        split_tensors = torch.split(v1, [1, 1, 1], dim=1)
        return (torch.cat(split_tensors, dim=1), torch.split(v1, [1, 1, 1], dim=1))
x1 = torch.randn(1, 3, 64, 64)
