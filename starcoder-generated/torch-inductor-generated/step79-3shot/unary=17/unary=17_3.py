
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose1d(3, 8, 3, padding=0, stride=1)
    def forward(self, x1):
        v1 = torch.conv_transpose1d(x1, 3, 8, 3, padding=0, stride=1)
        v2 = torch.relu(v1)
        v3 = torch.nn.Linear(8, 3)
        v4 = torch.sigmoid(v3)
        return v4
# Inputs to the model
x1 = torch.randn(1,3,8)
