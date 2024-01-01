
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(128, 4, (2, 2), stride=(1, 1)) # 4 and stride=(1, 1) are not from public PyTorch API
    def forward(self, x1):
        v1 = self.conv_transpose(x1)
        v2 = torch.tanh(v1)
        return v2
# Inputs to the model
x1 = torch.randn(1, 128, 32, 32)
input_to_model = (x1)
