
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(2, 2, 2)
        self.conv2 = torch.nn.Conv2d(2, 2, 2)
    def forward(self, x2):
        y = torch.nn.functional.batch_norm(self.conv1(x2), self.conv2(x2))
        return y
# Inputs to the model
x2 = torch.randn(1, 2, 4, 4)
