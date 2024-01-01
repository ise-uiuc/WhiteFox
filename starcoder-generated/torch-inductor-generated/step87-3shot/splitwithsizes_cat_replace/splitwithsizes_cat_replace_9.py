
class Block(torch.nn.Module):
    def __init__(self, inp, oup, hidden):
        super().__init__()
        self.inp = inp
        self.oup = oup
        self.hidden = hidden
        self.conv1 = torch.nn.Conv2d(self.inp, self.hidden, 1, 1, 0)
        self.block1 = torch.nn.ModuleList([torch.nn.Linear(self.hidden, self.hidden), torch.nn.ReLU()])
        self.conv2 = torch.nn.Conv2d(self.hidden, self.hidden, 1, 1, 0, bias=False)
        self.block2 = torch.nn.ModuleList([torch.nn.Linear(self.hidden, self.hidden), torch.nn.ReLU()])
        self.bn2 = torch.nn.BatchNorm2d(self.hidden, affine=False, track_running_stats=False)
        self.conv3 = torch.nn.Conv2d(self.hidden, self.oup, 1, 1, 0, bias=False)
        self.block3 = torch.nn.ModuleList([torch.nn.Linear(self.oup, self.oup), torch.nn.ReLU()])
    def forward(self, v1):
        split_tensors = torch.split(v1, [1, 1, 1], dim=1)
        concatenated_tensor = torch.cat(split_tensors, dim=1)
        return (concatenated_tensor, torch.split(v1, [1, 1, 1], dim=1))
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.net = torch.nn.Sequential(Block(3, 3, 16), torch.nn.Conv2d(3, 1, 1, 1, 0))
        self.features = torch.nn.Sequential(self.net, Block(1, 1, 16), self.net)
    def forward(self, split_tensors):
        split_tensors = torch.split(split_tensors, [1, 1, 1], dim=1)
        concatenated_tensor = torch.cat(split_tensors, dim=1)
        return (concatenated_tensor, torch.split(split_tensors, [1, 1, 1], dim=1))
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
