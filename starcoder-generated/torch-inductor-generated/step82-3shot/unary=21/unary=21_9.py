
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.convolution1d = torch.nn.Conv1d(5, 8, 2, stride=4, padding=1)
    def forward(self, x):
        v1 = self.convolution1d(x)
        v2 = torch.tanh(v1)
        return v2
# Inputs to the model
tensorInput = torch.randn(1, 5, 9)
