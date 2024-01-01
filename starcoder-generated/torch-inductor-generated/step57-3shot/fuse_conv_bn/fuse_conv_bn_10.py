
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv1d(2, 6, 1)
        self.conv2 = torch.nn.Conv1d(6, 6, 1)
    def forward(self, x):
        x = self.conv2(torch.tanh(torch.relu(self.conv1(x))))
        return x
# Inputs to the model
x = torch.randn(1, 2, 64)
