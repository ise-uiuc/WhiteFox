
class Model(torch.nn.Module):
     def __init__(self):
         super().__init__()
     def forward(self, x1, x2):
         v0 = x1.permute(0, 2, 1)
         v0 = x2.permute(0, 2, 1)
         v0 = x2.permute(0, 2, 1)
         v1 = v0.permute(0, 2, 1)
         v1 = v0.permute(0, 2, 1)
         v1 = v0.permute(0, 2, 1)
         v2 = torch.matmul(x1, v0)
         v2 = torch.bmm(x2, v1)
         v2 = x2.permute(0, 1, 2).permute(2, 1)
         v2 = torch.matmul(x1, v0.permute(0, 2, 1))
         v2 = v2.permute(0, 2, 1)
         v2 = v0.permute(0, 2, 1)
         v2 = v0.permute(0, 2, 1)
         return v2
# Inputs to the model
x1 = torch.randn(1, 2, 2)
x2 = torch.randn(1, 2, 2)
