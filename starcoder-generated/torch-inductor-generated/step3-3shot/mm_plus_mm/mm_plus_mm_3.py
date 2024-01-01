
class Model(torch.nn.Module):
    def forward(self, x1, x2, x3, x4):
        v1 = torch.mm(x1, x2)
        v2 = torch.mm(x3, x4)
        v3 = v1 + v2
        return v1
# Inputs to the model
x1 = torch.randn((4096,12), device = "cuda")
x2 = torch.randn((12,33), device = "cuda")
x3 = torch.randn((4096,12), device = "cuda")
x4 = torch.randn((12,37), device = "cuda")
