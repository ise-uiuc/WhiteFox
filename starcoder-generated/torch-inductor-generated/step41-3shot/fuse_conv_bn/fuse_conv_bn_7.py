
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        c = torch.nn.Conv2d(3, 32, kernel_size=1, padding=0)
        torch.manual_seed(5)
        c.weight = torch.nn.Parameter(torch.ones(c.weight.size()), requires_grad=False)
        self.conv1 = c
        self.relu1 = torch.nn.ReLU()
        self.conv2 = c
        self.relu2 = torch.nn.ReLU()
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        return self.relu2(x)
# Inputs to the model
x = torch.randn(4, 3, 3, 3)
