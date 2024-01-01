
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super(ModelTanh, self).__init__()
        self.conv = torch.nn.Conv2d(3, 16, (3, 7), 2)
        self.linear_1 = torch.nn.Linear(15*40, 10)
        self.linear_2 = torch.nn.Linear(10, 10)
        self.linear_3 = torch.nn.Linear(10, 2)
        self.tanh = torch.nn.Tanh()
    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.tanh(x)
        x = self.linear_1(x)
        x = x.tanh()
        x = self.linear_2(x)
        x = self.linear_3(x)
        return x
# Inputs to the model
x = torch.randn(1, 3, 64, 64)
