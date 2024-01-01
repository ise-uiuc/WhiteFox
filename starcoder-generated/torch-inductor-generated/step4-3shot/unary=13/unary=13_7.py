
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x1):
        x1 = x1.view(-1, 40)
        v1 = self.linear(x1)
        v2 = sigmoid(v1)
        v3 = v1 * v2
        return v3
