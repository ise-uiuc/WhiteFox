
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
    def forward(self, x):
        x = F.relu(x)
        return self.layer(x)
# Inputs to the model
x = torch.randn(2, 1, 4, 4)
