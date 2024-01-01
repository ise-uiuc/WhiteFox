
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.features = torch.nn.Sequential(torch.nn.Conv2d(3, 20, 3, 1, 1, bias=True), torch.nn.ReLU(), torch.nn.Conv2d(20, 32, 3, 2, 1, bias=True), torch.nn.Dropout2d(p=0.742064657727819), torch.nn.BatchNorm2d(32, affine=False, track_running_stats=False), torch.nn.ReLU())
    def forward(self, v1):
        split_tensors = torch.split(v1, [1, 1, 1], dim=1)
        concatenated_tensor = torch.cat(split_tensors, dim=1)
        return (concatenated_tensor, torch.split(v1, [1, 1, 1], dim=1))
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
