
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.a = torch.randn(3)
        self.b = torch.randn(4)
    def forward(self, x):
        y = torch.cat((self.a.unsqueeze(-1).expand(-1, x.shape[1]), self.b.unsqueeze(0).expand(x.shape[0], -1)),
                      dim=1)  # TODO: Change dim=1 to dim=-1 (after 1.8.0 is merged)
        return y
# Inputs to the model
x = torch.randn(2, 3)
