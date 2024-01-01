
class Model(torch.nn.Module):
    def forward(self, x, y):
        x = torch.cat((x, x, x, x), dim=1)
        x = x.view(-1, x.shape[1] * x.shape[2])
        z = torch.min(x, dim=1)[0]
        z = torch.relu(z)
        y = torch.cat((y, y), dim=1).view(y.shape[0], 15)
        y = y.view(y.shape[0], -1)
        y = torch.tanh(y)
        return z[:, None] * y + x * 2
# Inputs to the model
x = torch.randn(10, 2, 3, 4)
y = torch.randn(5, 15, dtype=torch.int64)
