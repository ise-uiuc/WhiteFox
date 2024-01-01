
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv1d(1, 9, 1, stride=1, padding=0)
        self.conv2 = torch.nn.Conv1d(9, 10, 1, stride=1, padding=0)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.conv2(v1)
        v3 = v2 + 5
        v4 = v3 - v1
        v5 = torch.nn.functional.tanh(v4)
        return v4
# Inputs to the model
x1 = torch.randn(1, 1, 30) 
