
class Model(torch.nn.Module):
    def __init__(self, min, max):
        super().__init__()
        return
    def forward(self, x1):
        v1 = torch.clamp_min(x1, min=0.5, out=torch.cuda.FloatTensor())
        v2 = torch.clamp_max(v1, max=0.5, out=torch.cuda.FloatTensor())
        return v2
# Inputs to the model
x1 = torch.randn(1, 28, 28)
