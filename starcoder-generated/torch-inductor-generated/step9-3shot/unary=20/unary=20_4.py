
class Model(torch.nn.Module):
    def __init__(self, ):
        super(Model, self).__init__()
        self.conv = torch.nn.Conv2d(16, 3, kernel_size=(1, 2), stride=(1, 1))
    def forward(self,x):
        x = self.conv(x)
        x = torch.sigmoid(x)
        return x
# Inputs to the model
x = torch.randn(1, 16, 5, 5)
