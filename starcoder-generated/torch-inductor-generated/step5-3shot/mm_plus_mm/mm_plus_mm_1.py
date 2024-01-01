
class Model(torch.nn.Module):
    def forward(self, x, y):
        self.y = y
        t1 = torch.cat((x, y), dim=0)
        t2 = torch.cat((x, y), dim=1)
        return t1, t2
# Inputs to the model
x = torch.randn(3, 8)
y = torch.randn(3, 8)
