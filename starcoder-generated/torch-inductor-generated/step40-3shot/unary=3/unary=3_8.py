
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv1d(2, 5, 7, stride=3, padding=3)
        self.conv2 = torch.nn.Conv1d(5, 3, 9, stride=1, padding=4)
    def forward(self, x1):
        v1 = self.conv1(x1) 
        v2 = v1 * 0.5
        v3 = self.conv2(v2) * 0.7071067811865476
        return v3
# Inputs to the model
x1 = torch.randn(1, 2, 4160)
