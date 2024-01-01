
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = torch.nn.ReLU()
        self.transpose = torch.nn.ConvTranspose1d(20, 1, 1, bias=False)
    def forward(self, x1):
        v1 = self.transpose(x1)
        v2 = self.relu(v1)
        return v2
# Inputs to the model
x1 = torch.randn(1, 20, 32)
