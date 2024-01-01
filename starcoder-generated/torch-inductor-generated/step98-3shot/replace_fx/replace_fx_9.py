
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weightA = 0.125
        self.tanh = torch.nn.Tanh()
    def forward(self, x, y):
        x1 = torch.rand_like(x, dtype=torch.double)
        x2 = x1.unsqueeze(-1)
        x3 = torch.bmm(x, self.tanh(y).mean(dim=-1).unsqueeze(-1).unsqueeze(-1).to(dtype=torch.float))
        x = x2 * x3
        x = x.mean(dim=-1).squeeze()
        return (x, )
# Inputs to the model
x = torch.randn(2, 2, 7, 7)
y = torch.randn(2, 17, 7, 7)
