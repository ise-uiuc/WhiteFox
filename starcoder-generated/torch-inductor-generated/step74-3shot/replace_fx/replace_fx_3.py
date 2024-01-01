
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        x2 = F.dropout(x1, p=0.2)
        x3 = torch.rand_like(x1, dtype=torch.float32, layout=torch.strided, device=torch.device('cpu'), requires_grad=False)
        return x3
# Inputs to the model
x1 = torch.randn(1)
