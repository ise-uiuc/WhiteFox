
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(5, 5, 6, stride=5, padding=0)
    def forward(self, x2):
        v1 = self.conv_transpose(x2) 
        v2 = torch.tanh(v1)
        return v2
# Inputs to the model
x2 = torch.randn(1, 5, 7, 7)
