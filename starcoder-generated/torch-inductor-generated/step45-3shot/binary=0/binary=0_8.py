
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 4, 1, stride=1, padding=1)
    def forward(self, x1, weight1=1, bias1=None, weight2=None, bias2=None):
        if bias1 == None:
            bias1 = torch.randn(self.conv1.weight.shape[0])
        if weight2 == None:
            weight2 = torch.randn(self.conv1.weight.shape[0] * 2, self.conv1.weight.shape[1] * 2, self.conv1.weight.shape[2], self.conv1.weight.shape[3])
        if bias2 == None:
            bias2 = torch.randn(self.conv1.weight.shape[0] * self.conv1.weight.shape[1])
        v1 = self.conv1(x1)
        return v1
# Inputs to the model
x1 = torch.randn(8, 3, 64, 64)
