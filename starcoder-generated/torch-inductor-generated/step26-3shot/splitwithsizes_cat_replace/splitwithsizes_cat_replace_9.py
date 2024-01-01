
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv2d = torch.nn.Sequential(torch.nn.AdaptiveAvgPool2d(7), torch.nn.AlphaDropout(0.0), torch.nn.Conv2d(3, 3, 1, 1, 0), torch.nn.ReLU6(), torch.nn.AdaptiveMaxPool2d(7), torch.nn.AdaptiveAvgPool2d((2, 2)), torch.nn.Conv2d(3, 3, 1, 1, 0), torch.nn.ReLU())
        self.relu = torch.nn.Sequential(torch.nn.AdaptiveAvgPool2d((2, 2)), torch.nn.Conv2d(3, 3, 1, 1, 0), torch.nn.ReLU(), torch.nn.Sigmoid(), torch.nn.ReLU(), torch.nn.BatchNorm2d(1), torch.nn.Conv2d(1, 1, 1, 1, 0), torch.nn.ReLU())
        self.conv2d1 = torch.nn.Sequential(torch.nn.Conv2d(3, 3, 1, 1, 0), torch.nn.Sigmoid(), torch.nn.BatchNorm2d(3), torch.nn.ReLU(), torch.nn.ReLU(), torch.nn.Softmax(dim=1))
        self.pad = torch.nn.Sequential(torch.nn.ConstantPad3d(1, value=3.964261))
    def forward(self, v1):
        split_tensors = torch.split(v1, [1, 1], dim=0)
        concatenated_tensor = torch.cat(split_tensors, dim=0)
        return (concatenated_tensor, torch.split(v1, [1, 2], dim=1))
# Inputs to the model
x = torch.randn(2, 3, 64, 64)
