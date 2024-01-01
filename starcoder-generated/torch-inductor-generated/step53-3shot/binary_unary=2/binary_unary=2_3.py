
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.softmax = torch.nn.Softmax(2)
    def forward(self, x1):
        v1 = self.softmax(x1, 1)
        v2 = v1 - 192
        v3 = F.relu(v2)
        v4 = torch.squeeze(v3, 0)
        return v4
# Inputs to the model
x1 = torch.randn(1, 192, 35, 45)
