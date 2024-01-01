
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2):
        t1 = torch.rand_like(x1, device=torch.device('cpu'), dtype=torch.float32, layout=torch.strided, pin_memory=False, requires_grad=True)
        return t1
# Inputs to the model
x1 = torch.randn(3, 4)
x2 = torch.randn(6, 7)
