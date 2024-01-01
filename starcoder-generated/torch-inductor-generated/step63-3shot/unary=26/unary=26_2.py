
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.input = torch.nn.ConvTranspose1d(38, 19, 2, stride=1, padding=1, bias=False, dilation=1, groups=1)
        self.relu = torch.nn.ReLU(inplace=False)
        self.add = torch.nn.Add()
    def forward(self, x111):
        y1 = self.input(x111)
        return self.add(self.relu(y1), torch.nn.functional.adaptive_avg_pool1d(y1, (10)))
# Inputs to the model
x111 = torch.randn(15, 38, 147)
