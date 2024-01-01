
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        x = torch.cat((torch.relu(x), torch.tanh(x)), dim=1)
        return x
# Inputs to the model
x = torch.randn(2, 2, 2)
