
# Re-create the model using another matrix multiplication pattern
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2, x3):
        v1 = torch.mm(y1, x2)
        v2 = torch.mm(y2, x3)
        v3 = v1 + v2
        return v3
# Inputs to the new model
y1 = torch.randn(3, 3)
y2 = torch.randn(3, 3)
x1 = torch.randn(3, 3, requires_grad=True)
x2 = torch.randn(3, 3)
x3 = torch.randn(3, 3)
