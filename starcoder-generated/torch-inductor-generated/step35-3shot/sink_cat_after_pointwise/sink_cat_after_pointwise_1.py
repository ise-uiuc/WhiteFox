
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        x = x.view(x.shape)
        x = x.type(torch.FloatTensor)
        x.relu()
        return x
# Inputs to the model
x = torch.randn(2, 3, 4)
