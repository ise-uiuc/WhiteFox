
class Model(nn.Module):
    def forward(self, x, y):
        x = torch.cat([x, x, x], dim=1)
        x = torch.cat([x, x], dim=1)
        y = torch.cat([y, y], dim=1)
        y = torch.cat([y, y, y], dim=1)
        return x, y


m = Model()
m(torch.randn(1, 2), torch.randn(2, 3))
