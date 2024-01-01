
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = torch.nn.SyncBatchNorm(8)
    def forward(self, x1):
        v1 = self.norm(x1)
        return v1
# Inputs to the model
x1 = torch.randn(8, 8, 4, 2)
