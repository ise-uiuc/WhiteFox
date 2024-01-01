
class m(torch.nn.Module):
    def forward(self, x):
        t2 = torch.rand_like(x, dtype = torch.float16, device = torch.device('cuda'), layout = torch.strided, memory_format = torch.preserve_format)
        return t2
# Inputs to the model
x = torch.randn(1)
