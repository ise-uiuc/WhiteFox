
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        y1 = torch.cat([x, x], dim=1)
        y2 = y1.view(y1.shape[0], y1.shape[1] * y1.shape[2] * y1.shape[3])
        y3 = torch.relu(y2)
        y4 = y3 if y3.shape == (3, 16) else y1.permute(0, 2, 3, 1)
        return y4
# Inputs to the model
x = torch.randn(2, 1, 2, 4)
