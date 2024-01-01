
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = torch.nn.ReLU(inplace=False)
        self.conv1d = torch.nn.Conv1d(16, 64, 1, bias=False)
    def forward(self, x):
        t1 = self.relu(x)
        t2 = self.conv1d(t1)
        t3 = torch.tanh(t2)
        y1 = nn.Sigmoid(t3)
        return y1
# Inputs to the model
x = torch.randn(128, 16, 40)
