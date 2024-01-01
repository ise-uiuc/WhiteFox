
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 3, stride=1, padding=1)
 
    def forward(self, x):
        v1 = self.conv(x)
        v2 = torch.softmax(F.dropout(torch.matmul(v1, v1.transpose(-2, -1) * 0.7071067811865476), 0.9), dim=-1)
        res = torch.matmul(v2, v1)
        return res

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(1, 3, 64, 64)
