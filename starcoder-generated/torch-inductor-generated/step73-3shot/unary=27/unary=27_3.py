
class Model(torch.nn.Module):
    def __init__(self, weight=torch.nn.Parameter(torch.Tensor([0.5]))):
        super().__init__()
        self.weight = weight
    def forward(self, x1):
        v1 = torch.clamp_min(x1, self.weight)
        return v1
# Inputs to the model
x1 = torch.randn(2,)
