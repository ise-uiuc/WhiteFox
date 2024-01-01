
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.test = 0
    def forward(self, x):
        x = self.somemethod(x)
        x = torch.cat((x, x), dim=1)
        if x.dim() == 3:
            x = x.tanh()
        else:
            x = x.view(-1).tanh()
        return x

    def somemethod(self, x):
        return self.othermethod(x)

    def othermethod(self, x):
        x_relu = torch.relu(x)
        x_tanh = torch.tanh(x)
        x_sigmoid = torch.sigmoid(x)
        out = x_relu * x_tanh - 2.0 * x_sigmoid
        return out
# Inputs to the model
x = torch.randn(2, 3, 4)
