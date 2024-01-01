
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv2d = torch.nn.Conv2d(3,20,1,stride=1,padding=1,stride=1)
        self.tanh = torch.nn.Tanh()

    def forward(self, x):
        x = self.conv2d(x)
        x = self.tanh(x)
        return x
# Inputs to the model
x = torch.randn(128,3,224,224)
