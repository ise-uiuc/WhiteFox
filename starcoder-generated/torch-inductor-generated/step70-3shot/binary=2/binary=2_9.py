
class Model(torch.nn.Module):
    def __init__(self, ):
        super().__init__()
        self.op = torch.ops.myop.myop
    def forward(self, x):
        return self.op(x) - 200000000000
# Inputs to the model
x = torch.randn(2, 3, 4)
