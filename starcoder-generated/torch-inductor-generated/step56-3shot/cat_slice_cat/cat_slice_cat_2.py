
class Model(torch.nn.Module):
    def forward(self, x1, x2, x3) -> torch.Tensor:
        y = torch.cat([x1, x2, x3], dim=1)
        x = [y]
        shape = y.shape
        for i in shape:
            x += [torch.arange(i)]
        x = torch.stack(x)
        x = torch.roll(x, 1, 1)
        return x

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(2, 3, 4, 5)
x2 = torch.randn(2, 4, 4, 5)
x3 = torch.randn(2, 5, 4, 5)
