
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        x2 = lowmem_dropout(x1, p=0.5, memory_format=torch.channels_last)
        x3 = torch.rand_like(x1, memory_format=torch.channels_last)
        x4 = torch.randn(1)
        x5 = torch.sum(x4)
        return x5
# Inputs to the model
x1 = torch.randn(5, 5, 16, 16, device='cuda', dtype=torch.float32)
