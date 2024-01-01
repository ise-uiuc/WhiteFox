
class SinkTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = None
        self.in_features = 2
        self.out_features = 4
    def forward(self, x):
        x = torch.cat([x, x, x, x], dim=0).view(1, -1, 4)
        x = torch.tanh(x.squeeze(dim=0)).permute(1, 0, 2)
        return x
# Inputs to the model
x = torch.randn(3, 2, requires_grad=True)
