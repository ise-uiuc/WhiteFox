
class ModelTanh(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 5, kernel_size=(1, 1))
 
    def forward(self, x):
        x = self.conv1(x)
        x = torch.tanh(x)
        return x
# Inputs to the model
x = torch.randn(1, 1, 224, 224)
