
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.features = torch.nn.Sequential(torch.nn.Dropout(p=0.25), torch.nn.BatchNorm2d(num_features=8, momentum=0.2, eps=1e-05, affine=True, track_running_stats=True), torch.nn.Hardtanh(min_val=0, max_val=6, inplace=False))
    def forward(self, v1):
        split_tensors = torch.split(v1, [1, 1, 1], dim=1)
        concatenated_tensor = torch.cat(split_tensors, dim=1)
        return (concatenated_tensor, torch.split(v1, [1, 1, 1], dim=1), torch.split(v1, [1, 1, 1], dim=1))
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
