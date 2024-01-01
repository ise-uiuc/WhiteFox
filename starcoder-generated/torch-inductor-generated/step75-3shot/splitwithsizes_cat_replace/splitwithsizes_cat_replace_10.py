
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.features = torch.nn.Sequential(torch.nn.Conv2d(32, 32, 8, 8, 0), torch.nn.BatchNorm2d(32), torch.nn.ReLU(inplace=False), torch.nn.Conv2d(32, 8, 1, 1, 0))
    def forward(self, v1):
        v2 = v1.contiguous()
        split_tensors = torch.split(v1, [1, 1, 1], dim=1)
        concatenated_tensor = torch.cat(split_tensors, dim=1)
        return (concatenated_tensor, torch.split(v1, [1, 1, 1], dim=1))
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
