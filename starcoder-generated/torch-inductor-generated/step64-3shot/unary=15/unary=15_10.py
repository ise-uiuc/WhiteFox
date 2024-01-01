
class S2_Flatten_Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv1d(50, 5, 5, stride=1, padding=0)
        self.conv2 = torch.nn.Conv1d(5, 20, 1, stride=1, padding=0)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = torch.flatten(v1)
        v3 = self.conv2(v2)
        v4 = torch.flatten(v3)
        return v4
# Inputs to the model
x1 = torch.randn(1, 50, 60)
