
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        for i in range(0, x1.size(0)):
            x1 = torch.cat([x1.view(-1)[x1.view(-1) > 0.0].view(-1, 1), x1.view(-1)[torch.logical_and(x1.view(-1) > 0.0, x1.view(-1) > 0.0)].view(-1, 1), x1.view(-1)[x1.view(-1) > 0.0].view(-1, 1)], 0)
        x1.view(-1)[x1.view(-1) > 0.0].view(-1, 1)
        x1.view(-1)[torch.logical_and(x1.view(-1) > 0.0, x1.view(-1) > 0.0)].view(-1, 1)
        x1.view(-1)[x1.view(-1) > 0.0].view(-1, 1)
        return x1
# Inputs to the model
x1 = torch.randn(5, 5)
