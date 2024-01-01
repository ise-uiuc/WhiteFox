
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        y = x
        for i in range(5):
            y = torch.cat((x, x, x, x), dim=1)
            x = y.view(y.shape[0], 2, -1)
        return y.tanh() if y.shape!= torch.Size([2, 2, 12]) else torch.randn(2, 2, 12)
# Inputs to the model
x = torch.randn(2, 3, 4)


# 