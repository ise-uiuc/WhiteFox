
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
        self.normal = torch.distributions.normal.Normal(0, 2)
        self.sigmoid = torch.nn.ReLU()
    def forward(self, x):
        g1 = x.permute(0, 2, 1)
        g2 = torch.nn.functional.linear(g1, self.linear.weight, self.linear.bias)
        g3 = g2.permute(0, 2, 1)
        g4 = torch.nn.functional.hardtanh(g3)
        g5 = g3.permute(0, 2, 1)
        g6 = torch.nn.functional.interpolate(g5, scales_t=1, mode="linear", align_corners=False)
        g7 = self.linear(g6)
        g8 = g7 + 2
        g9 = torch.nn.functional.relu(g8)
        return g9
# Inputs to the model
x1 = torch.randn(1, 2, 2)
