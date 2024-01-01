
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.features = torch.nn.BatchNorm2d(32, affine=False, track_running_stats=False)
    def forward(self, v1, v2):
        split_tensors = torch.split(v1, [1, 1, 1], dim=1)
        concatenated_tensor = torch.cat([split_tensors[i] for i in range(len(split_tensors))], dim=1) # This concatenation operation will trigger the pattern.
        return (concatenated_tensor, torch.split(v2, [1, 1, 1], dim=1))
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
