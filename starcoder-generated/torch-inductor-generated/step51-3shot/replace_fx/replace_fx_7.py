
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv1d(768, 512, 24)
        self.layer = torch.nn.LayerNorm(512)
        self.drop = torch.nn.Dropout(0.1)
    def forward(self, x1):
        x2 = self.conv1(x1)
        x3 = self.layer(x2)
        x4 = self.drop(x3)
        x5 = self.drop(x1)
        x6 = torch.rand_like(x5)
        return x6
# Inputs to the model
x1 = torch.randn(8, 768, 512)
