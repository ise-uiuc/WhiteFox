
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        y = torch.cat((torch.ones((1, 2), requires_grad=True), torch.ones((1, 2), requires_grad=True)), dim=1)
        return y.relu()
# Inputs to the model
x = torch.randn(2, 3)
