
class Model(torch.nn.Module):
    def __init__(self, inp, hidden, out):
        super(Model, self).__init__()
        self.features = torch.nn.Sequential(torch.nn.Conv2d(inp, hidden, kernel_size=3, stride=1, padding=0, bias=False), torch.nn.ReLU(inplace=False), torch.nn.AvgPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=False), torch.nn.BatchNorm2d(hidden, affine=False, track_running_stats=False), torch.nn.Conv2d(hidden, out, kernel_size=3, stride=1, padding=0, bias=False), torch.nn.Conv2d(hidden, out, kernel_size=5, stride=1, padding=2, bias=False))
    def forward(self, v1):
        split_tensors = torch.split(v1, [1, 1, 1, 1, 1, 1], dim=1)
        concatenated_tensor = torch.cat(split_tensors, dim=1)
        return (concatenated_tensor, torch.split(v1, [1, 1, 1, 1, 1, 1], dim=1))
# Inputs to the model
x1 = torch.randn(1, 3, 112, 112)
