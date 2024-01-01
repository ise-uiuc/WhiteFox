
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = torch.nn.Conv1d(3, 3, 2)
    def forward(self, x1):
        v1 = torch.mm(x1, x1) + self.emb(x1)
        return v1
# Inputs to the model
x1 = torch.randn(3, 3)
