
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(1, 1, padding_idx=1)
    def forward(self, x1):
        v1 = x1.int()
        v2 = self.embedding(v1)
        v3 = v2.squeeze(1)
        return v3
# Inputs to the model
x1 = torch.tensor([0])
