
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.split = torch.nn.Parameter(torch.randn(3, 32, 3, 3))
        self.block = [torch.nn.Conv2d(3, 32, 3, 1, 1, bias=False)]
    def forward(self, v1):
        split_tensors = torch.split(v1, [1, 1, 1], dim=1)
        concatenated_tensor = torch.cat(split_tensors, dim=1)
        return (concatenated_tensor, torch.split(v1, [1, 1, 1], dim=1))
x1 = torch.randn(1, 3, 64, 64)
