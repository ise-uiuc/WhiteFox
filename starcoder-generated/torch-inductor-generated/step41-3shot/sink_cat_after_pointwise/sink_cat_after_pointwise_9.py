
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        x1 = x.view(x.shape + (1,))
        x2 = torch.transpose(x1, 0, 2)
        x3 = torch.cat((x2, x2), dim=0)
        x4 = torch.cat((x1, x3), dim=1)
        x5 = x4.sum(dim=2)
        # This is a dummy pointwise operation in order to get a ReLU after concatenation.
        x6 = torch.cat((x5, x5), dim=0)
        return 2.0 * x6
# Inputs to the model
x = torch.randn(2, 3, 2, 2)
