
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Linear(2, 4)
    def forward(self, x):
        x = self.layers(x)
        tensor1 = torch.addmm(x, 1.0, 1.0)
        tensor2 = torch.cat((tensor1, tensor1), dim=1)
        return tensor2
# Inputs to the model
x = torch.randn(2, 2)
