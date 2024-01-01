
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        y1 = torch.rand_like(x1, dtype=torch.int, layout=torch.strided, device=x1.device, pin_memory=True, requires_grad=False, memory_format=torch.contiguous_format)
        b1 = F.dropout(x1, p=0.2)
        return y1
# Inputs to the model
x1 = torch.randint(5, (1, 2), dtype=torch.float16)
