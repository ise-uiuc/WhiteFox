
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        x2 = F.log_softmax(x1)
        x3 = F.log_softmax(x2, dim=1)
        t1 = torch.softmax(x3, dim=1)
        return x2, t1
# Inputs to the model
x1 = torch.randn(10, 5)
