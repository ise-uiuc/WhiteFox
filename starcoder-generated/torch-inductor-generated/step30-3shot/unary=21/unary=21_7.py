
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super(ModelTanh, self).__init__()
        self.conv = torch.nn.Conv2d(3, 1, 3)
        self.relu = torch.nn.ReLU()
        self.max_pool = torch.nn.MaxPool2d(3, 2)
        self.tanh = torch.nn.Tanh()
        self.avg_pool = torch.nn.AvgPool2d(3, 2)
        self.flatten = torch.nn.Flatten()
        self.linear = torch.nn.Linear(225, 1010)
        self.relu_a = torch.nn.ReLU()
        self.linear_1 = torch.nn.Linear(1010, 410)
        self.relu_b = torch.nn.ReLU()
        self.linear_e = torch.nn.Linear(410, 26250)
    def forward(self, x):
        r1 = self.conv(x)
        r2 = self.relu(r1)
        r3 = self.max_pool(r2)
        r4 = self.tanh(r3)
        r5 = self.avg_pool(x)
        r6 = self.flatten(r4)
        r7 = self.linear(r5)
        r8 = self.relu_a(r7)
        r9 = self.linear_1(r8)
        r10 = self.relu_b(r9)
        r11 = self.linear_e(r10)
        return r11
# Inputs to the model
x1 = torch.randn(1, 3, 32, 34)
