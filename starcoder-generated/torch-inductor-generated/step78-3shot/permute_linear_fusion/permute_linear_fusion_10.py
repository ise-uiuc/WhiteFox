
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
        self.sigmoid = torch.nn.Softmax(-1)
    def forward(self, x):
        g1 = x.permute(0, 2, 1)
        g2 = torch.nn.functional.linear(g1, self.linear.weight, self.linear.bias)
        g3 = self.sigmoid(g2)
        g4 = g3 * g1
        g5 = torch.tensor([1.1, 0.7, 0.1])
        g6 = g6.permute(0, 2, 1)
        g7 = torch.nn.functional.linear(g6, g5, 0)
        g8 = g7.permute(0, 2, 1)
        return g8
# Inputs to the model
x1 = torch.randn(1, 2, 2)
