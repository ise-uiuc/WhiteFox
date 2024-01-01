
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = torch.nn.TransformerEncoder()
        self.layer2 = torch.nn.TransformerEncoderLayer(20, 5)
    def forward(self, x0):
        x = self.layer1(x0)
        y = self.layer2(x)
        return y
# Inputs to the model
x0 = torch.randn(4, 6, 20)
x1 = torch.randn(7, 6, 20)
