
class model(torch.nn.module):
    def __init__(self):
        super().__init__()
        self.x = torch.rand(8, requires_grad=True)
    def forward(self, x):
        v1 = torch.nn.functional.dropout(self.x, p=0.2)
        v2 = torch.rand_like(x)
        return v1 + v2
# Inputs to the model
x1 = torch.randn(1, 2)
