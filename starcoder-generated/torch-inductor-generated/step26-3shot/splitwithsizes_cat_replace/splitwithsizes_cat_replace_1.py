
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.features = torch.nn.Sequential(torch.nn.Conv2d(3, 24, 5, 1, 2), torch.nn.Conv2d(24, 48, 1, 1, 0), torch.nn.Conv2d(24, 48, 1, 1, 0))
        if True:
            self.pad = torch.nn.Sequential(torch.nn.ConstantPad2d([0, 0, 1, 1], value=0.422196), torch.nn.PixelShuffle(2))
        self.res = torch.nn.Sequential(torch.nn.Conv2d(48, 48, 3, 2, 0, dilation=1, groups=24), torch.nn.ReLU(), torch.nn.Conv2d(48, 192, 1, 1, 0), torch.nn.Conv2d(192, 192, 1, 1, 0))
        self.relu = torch.nn.Sequential(torch.nn.ReLU(inplace=False), torch.nn.Conv2d(3, 192, 1, 1, 0), torch.nn.MaxPool2d(3, 2, 1), torch.nn.AdaptiveAvgPool2d(7), torch.nn.ReLU(inplace=False), torch.nn.Conv2d(192, 1000, 1, 1, 0))
    def forward(self, v1):
        split_tensors = torch.split(v1, [1, 1, 1], dim=0)
        concatenated_tensor = torch.cat(split_tensors, dim=0)
        return (concatenated_tensor, torch.split(v1, [1, 1, 1], dim=0))
# Inputs to the model
v1 = torch.Tensor(3, 47, 224, 1024)
