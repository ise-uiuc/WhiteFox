
class Block(torch.nn.Module):
    def __init__(self, inp):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(inp, 32, 3, 1, 1, groups=inp, bias=False)
        self.bn1 = torch.nn.BatchNorm2d(32, affine=False, track_running_stats=False)
        self.conv2 = torch.nn.Conv2d(32, inp, 1, 1, 1, bias=False)
        self.bn2 = torch.nn.BatchNorm2d(inp, affine=False, track_running_stats=False)
        self.fc = torch.nn.Linear(8192, inp)
        self.fc2 = torch.nn.Linear(inp, 8192)
    def forward(self, x):
        out = torch.nn.ReLU()(self.bn1(self.conv1(x)) + x)
        out = torch.nn.ReLU()(self.bn2(self.conv2(out)) + out)
        split_tensors = torch.split(out, [1, 1, 1], dim=1)
        concatenated_tensor = torch.cat(split_tensors, dim=1)
        out = self.fc(concatenated_tensor.view(concatenated_tensor.size(0), -1)) + self.fc2(out.view(out.size(0), -1))
        out = out.view((out.size(0), 8192, 1, 1))
        return out
class Model(torch.nn.Module):
    def __init__(self, inp):
        super().__init__()
        self.features = Block(inp)
        self.extra = torch.nn.ReLU()
    def forward(self, v1):
        split_tensors = torch.split(v1, [1, 1, 1], dim=1)
        concatenated_tensor = torch.cat(split_tensors, dim=1)
        return (concatenated_tensor, torch.split(v1, [1, 1, 1], dim=1))
# Inputs to the model
x1 = torch.randn(1, 3, 2, 2)
