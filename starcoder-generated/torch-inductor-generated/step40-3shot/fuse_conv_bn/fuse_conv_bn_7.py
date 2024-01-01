
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(1)
        conv_bn = torch.nn.Conv2d(5, 16, (1, 1), (1, 1), (0, 0), bias=False)
        self.model = torch.nn.Sequential(conv_bn)
    def forward(self, x):
        return self.model(x)
# Inputs to the model
x1 = torch.randn(1, 5, 5, 5)
