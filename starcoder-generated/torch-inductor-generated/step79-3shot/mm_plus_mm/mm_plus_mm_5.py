
class Model(nn.Module):
    def forward(self, x):
        y = torch.mm(x, x)
        v1 = torch.mm(x, x)
        return v1 + y
# Inputs to the model
x = torch.randn(16, 16).cuda()
