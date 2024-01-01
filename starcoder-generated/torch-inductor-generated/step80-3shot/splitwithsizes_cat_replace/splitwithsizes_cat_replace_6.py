
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 3, (3, 3), (1, 1), (0, 0), bias=False)
        self.conv2 = torch.nn.Conv2d(3, 3, (9, 9), (2, 2), (0, 0), groups=3, bias=True)
    def forward(self, x):
        split_tensors = torch.split(x, [1, 1, 1, 1], dim=1)
        out = [self.conv1(input) for input in split_tensors]
        out = torch.split(torch.cat(out, dim=1), [1, 1], dim=1)
        out = []
        for i in range(len(split_tensors)):
            out.append(self.conv2(split_tensors[i] + out[i]))
        return out
# Inputs to the model
x1 = torch.randn(1, 6, 64, 64)
