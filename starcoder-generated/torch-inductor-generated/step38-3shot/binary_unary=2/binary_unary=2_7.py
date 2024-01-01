
class Model(torch.nn.Module):
    def forward(self, x1):        
        v2 = x1 - 100
        v1 = torch.transpose(x1, 0, 2)
        v3 = v1 - v2
        return v3
# Inputs to the model
x1 = torch.randn(1, 8, 64, 64)
