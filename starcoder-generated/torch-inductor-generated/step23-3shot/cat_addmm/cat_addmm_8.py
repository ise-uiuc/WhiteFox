
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers1 = nn.Conv1d(10, 256, kernel_size=1)
        self.layers2 = nn.Conv2d(512, 1024, kernel_size=1)
    def forward(self, x):
        x = self.layers2(self.layers1(x))
        x = x.flatten(start_dim=2)
        x = x.flatten(start_dim=1)
        return x
# Inputs to the model
x = torch.randn(2, 16, 10, 10)
