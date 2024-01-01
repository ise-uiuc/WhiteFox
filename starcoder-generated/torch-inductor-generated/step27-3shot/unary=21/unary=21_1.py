
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv1d(kernel_size=2, stride=1, padding=1, bias=False)
        self.conv2 = torch.nn.Conv1d(kernel_size=2, stride=1, padding=1, bias=False)
    def forward(self, x):
        v1 = self.conv1(x)
        v2 = torch.tanh(v1)
        v3 = v2.transpose(1,2) # Transpose
        v4 = self.conv2(v3)
        v5 = torch.tanh(v4)
        v6 = v5.transpose(1,2)
        return v6
# Inputs to the model
x = torch.rand(1, 16, 10)
