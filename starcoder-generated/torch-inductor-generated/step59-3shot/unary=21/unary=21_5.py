
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv1d(1, 1, 5)
        self.conv2 = torch.nn.Conv1d(1, 1, 5)
    def forward(self, x7):
        h0 = self.conv1(x7)
        h0 = torch.tanh(h0)
        h1 = self.conv2(h0)
        h1 = torch.tanh(h1)
        return h1
# Inputs to the model
x7 = torch.randn(1, 1, 16)
