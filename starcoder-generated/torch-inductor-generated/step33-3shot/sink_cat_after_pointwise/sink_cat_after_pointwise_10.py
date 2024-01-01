
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        if (x.shape[1] == 2):
            y = torch.relu(x)
            z = torch.tanh(y)
        else:
            y1 = torch.tanh(x)
            y2 = torch.relu(x)
            z = torch.cat((y1, y2), dim=0)
            z = z.view(z.shape[0], -1).softmax(dim=1)
        return z
# Inputs to the model
x = torch.randn(2, 2, 2)
