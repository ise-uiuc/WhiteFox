
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Sequential(
            nn.Conv2d(10, 10, kernel_size=1),
            nn.LayerNorm([10, 10]),
            nn.Tanh()
        )
    def forward(self, x):
        output = self.layer(x)
        return output
# Inputs to the model
x = torch.randn(1, 10, 10, 10)
