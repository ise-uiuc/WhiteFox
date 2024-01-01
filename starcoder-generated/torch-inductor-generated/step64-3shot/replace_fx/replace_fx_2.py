
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.norm0 = torch.nn.LayerNorm([10, 30])
        self.norm1 = torch.nn.BatchNorm2d(10, affine=True)

    def forward(self, x):
        x = self.norm0(x)
        x = self.norm1(x)
        return x
# Inputs to the model
x = torch.randn([1, 10, 30])
