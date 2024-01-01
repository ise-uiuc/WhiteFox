
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = torch.nn.TransformerEncoderLayer(2, 2)
    def forward(self, x):
        v1 = self.layer(x)
        return v1
# Inputs to the model
x = torch.randn(2, 2, 2, device='cpu')
