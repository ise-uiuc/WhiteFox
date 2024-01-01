
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 3, 2)
    def forward(self, x):
        y = F.batch_norm(self.conv(x), running_mean=0.0, running_var=0.1, training=False)
        return y
# Inputs to the model
x = torch.randn(1, 3, 4, 4)
