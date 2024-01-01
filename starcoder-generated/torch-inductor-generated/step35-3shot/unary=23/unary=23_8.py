
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose1 = torch.nn.ConvTranspose2d(18, 342, kernel_size=(7, 3), stride=(1, 1), padding=(3, 1))
        self.conv_transpose2= torch.nn.ConvTranspose2d(342, 69, kernel_size=(7, 9), stride=(1, 1), padding=(3, 4))
    def forward(self, x1):
        v1 = self.conv_transpose1(x1)
        v2 = self.conv_transpose2(v1)
        v3 = torch.tanh(v2)
        return v3
# Inputs to the model
x1 = torch.randn(1, 18, 28, 28) # Input size 28x28
