
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.features = torch.nn.Sequential(torch.nn.Conv2d(2, 1, 3, 1, 1, bias=True), torch.nn.Hardtanh(inplace=True), torch.nn.BatchNorm2d(32, eps=0.01, momentum=0.1, affine=True, track_running_stats=True), torch.nn.Conv2d(32, 32, 3, 1, 1, groups=32, bias=True), torch.nn.ReLU(), torch.nn.AdaptiveMaxPool2d(output_size=(4, 4)))
    def forward(self, v1):
        split_tensors = torch.split(v1, [1, 1, 1], dim=1)
        concatenated_tensor = torch.cat(split_tensors, dim=1)
        return (concatenated_tensor, torch.split(v1, [1, 1, 1], dim=1))
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
