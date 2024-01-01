
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        lstm1 = torch.nn.LSTMCell(2, 5)
        v1 = lstm1(x1)
        v2 = torch.reshape(v1, (5,))
        v3 = v2.mean()
        v2 = v2.add(v3)
        return v1
# Inputs to the model
x1 = torch.randn(1, 2, 5)
