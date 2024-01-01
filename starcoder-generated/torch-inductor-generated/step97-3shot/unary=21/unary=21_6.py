
# No documentation on the torch.max() function
# No documentation on the torch.cat() function
# No documentation on the torch.ones() function
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(16, 4, 3, stride=2, padding=1)
        self.conv2 = torch.nn.Conv2d(4, 2, 3, stride=2, padding=1)
        self.conv3 = torch.nn.Conv2d(2, 1, 3, stride=2, padding=1)
        self.a_3 = torch.nn.AvgPool2d(3, stride=2, padding=1)
        self.a = torch.nn.AvgPool2d(3, stride=2, padding=1)
        self.x = torch.ones(1, 752)
    def forward(self, x0):
        a_1 = self.conv1(x0)
        a_2 = torch.tanh(a_1)
        a_4 = torch.tanh(self.conv2(a_2))
        a_5 = a_4
        a_6, b = torch.max(a_5, 1)
        a_7 = torch.tanh(torch.cat(a_6, b))
        a_8 = torch.cat(self.a_3(a_5), self.a(x0))
        a_0 = torch.cat(a_7, self.conv3(x0))
        a_9 = self.x
        v1 = a_8 + a_0 + a_9
        v2 = torch.tanh(v1)
        v3 = torch.tanh(torch.tanh(v2))
        return v3[0, :], v1, v2, v3
# Inputs to the model
x = torch.randn(1, 16, 3, 4)
