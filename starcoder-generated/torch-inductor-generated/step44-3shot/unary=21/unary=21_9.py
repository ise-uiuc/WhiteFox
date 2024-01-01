
class ModelTanh(torch.nn.Module):
    def __init__(self, num_classes=10, kernel_size=3):
        super(ModelTanh, self).__init__()
        self.conv1 = nn.Conv2d(3, num_classes, kernel_size=kernel_size, padding=(kernel_size-1)//2)
    def forward(self, inp):
        x = torch.tanh(inp)
        x = self.conv1(x)
        x = torch.tanh(x)
        return x

model = ModelTanh()
# Inputs to the model (must match input shape of model)
inp = torch.randn(32, 3, 224, 224)
