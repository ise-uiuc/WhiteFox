
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        y = x.view(x.shape[0], -1)
        y.expand(y.shape[0], y.shape[1])
        y = y.transpose(y.shape[0], y.shape[1])
        y = y.transpose(y.shape[0], y.shape[1])
        x = y.flatten(2) + y.flatten(2).transpose(y.shape[2], y.shape[3])
        x = x.view(x.shape[0], -1)
        return x
# Inputs to the model
x = torch.randn(2, 3, 4, 5)
# Model end

# Model begins
class Model(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        x = x.view(x.shape[0], 25)
        if x.shape[0] == 1:
            x = x.expand(2, 25)
        else:
            x = x.reshape(2, 25)
        x = x.contiguous()
        if x.is_contiguous():
            x = x.detach()
        else:
            x = x.contiguous()
        x = x.view(25)
        return x
# Inputs to the model
x = torch.randn(2, 3, 5)
# Model end
