
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(384, 768, kernel_size=(1, 1), stride=(1, 1), bias=True)
        self.conv2 = torch.nn.Conv2d(768, 128, kernel_size=(1, 1), stride=(1, 1), bias=True)
    def forward(self, x):
        t1 = self.conv1(x)
        t2 = torch.tanh(t1)
        t3 = self.conv2(t2)
        t4 = torch.tanh(t3)
        return t4
# Inputs to the model
x = torch.randn(1, 384, 35, 35)
