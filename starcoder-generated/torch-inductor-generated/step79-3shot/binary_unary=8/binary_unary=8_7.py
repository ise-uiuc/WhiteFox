
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        pass
    def forward(self, x1):
        t1 = x1 + x1
        t2 = torch.relu(t1)
        return t2
# Inputs to the model
x1 = torch.randn(4, 4, 224, 224)
