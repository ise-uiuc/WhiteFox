
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        y1 = torch.rand_like(x1, dtype=torch.float64, layout=torch.strided, device=x1.device, pin_memory=True, requires_grad=False, memory_format=torch.contiguous_format)
        c1 = F.dropout(y1, p=0.8)
        c2 = F.dropout(c1, p=0.5)
        c3 = F.dropout(c1)
        c4 = F.dropout(c2)
        return (c2, c3, c4)
x1 = torch.randn(1, 3)
