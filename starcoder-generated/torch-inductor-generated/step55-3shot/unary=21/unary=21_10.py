
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv1d(2, 2, 2, 6, 1)
        self.conv2 = torch.nn.Conv1d(2, 2, 2, 6, 1) 
    def forward(self, x):
        v1 = self.conv1(x)
        v2 = torch.tanh(v1)
        v3 = self.conv2(v2)
        v3 = torch.tanh(v2)
        v3 = torch.tanh(v2)
        return v3
# Inputs to the model
x = torch.randn(1, 2, 224)
