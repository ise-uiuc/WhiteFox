
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(2,10,kernel_size=3,stride=3)
        self.dropout = torch.nn.Dropout(p=0.5)
    def forward(self, x1):
        v1 = self.conv_transpose(x1)
        v2 = self.dropout(x1)
        v3 = torch.tanh(v2)
        return v3
# Inputs to the model
x1 = torch.randn(1, 2, 64, 64)
