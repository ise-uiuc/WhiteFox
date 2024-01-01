
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.features = torch.nn.BatchNorm2d(32, affine=False, track_running_stats=True)
    def forward(self, v0):
        split_tensors = torch.split(v0, [1, 1, 1], dim=1)
        split_tensor_1 = split_tensors[0]
        return (torch.cat([v0, split_tensor_1]), torch.split(v0, [1, 1, 1], dim=1))
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
