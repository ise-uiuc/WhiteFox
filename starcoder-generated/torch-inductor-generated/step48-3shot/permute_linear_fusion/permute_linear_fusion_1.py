
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(5, 10)
        self.softmax = torch.nn.Softmax(dim=0)
    def forward(self, x1):
        v1 = x1.permute(0, 2, 1)
        v2 = self.embedding(v1)
        v3 = self.softmax(v2)
        return v3
# Inputs to the model
x1 = torch.randint(5, (1, 2, 2))
