
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.features = torch.nn.BatchNorm2d(64, affine=False, track_running_stats=False)
    def forward(self, v1):
        split_tensors1 = torch.split(v1, [1, 1, 1, 1, 1, 1], dim=1)
        split_tensors2 = torch.split(torch.cat([split_tensors1[i] for i in range(len(split_tensors1))], dim=1), [1, 1], dim=1)
        concatenated_tensor1 = torch.cat([split_tensors2[i] for i in range(len(split_tensors2))], dim=1)
        return concatenated_tensor1
# Inputs to the model
x1 = torch.randn(1, 6, 64, 64)
