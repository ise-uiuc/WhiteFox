
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv1d(3, 3, kernel_size=1)
    def forward(self, x):
        t1 = self.conv1(x)
        t2 = t1 - 10
        return (t1, t2)
# Inputs to the model
x = torch.randn(1, 3, 16)
