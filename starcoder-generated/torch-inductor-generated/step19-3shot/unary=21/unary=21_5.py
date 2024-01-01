
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv=torch.nn.Conv2d(3,4,kernel_size=1,padding=2)
    def forward(self,x2):
        v3=self.conv(x2)
        v4=torch.tanh(v3)
        return v4
# Inputs to the model
x2 = torch.randn(1, 3, 64, 64)
