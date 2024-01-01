
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.features = torch.nn.Sequential(torch.nn.Conv2d(3, 16, 5, 1, 2), torch.nn.BatchNorm2d(16, affine=False, track_running_stats=True))
    def forward(self, v1):
        split_tensors = torch.split(v1, 1, dim=1)
        concatenated_tensor = torch.cat([split_tensors[0], split_tensors[1], split_tensors[2], split_tensors[3], split_tensors[4], split_tensors[4]], dim=1)
        return (concatenated_tensor, torch.split(v1, 1, dim=1))
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
