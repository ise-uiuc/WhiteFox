
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = torch.nn.Embedding(32, 16)
        self.dropout = torch.nn.Dropout(0.5)

    def forward(self, x1, x2):
        v1 = self.embed(x1)
        v2 = self.embed(x2)
        v3 = torch.matmul(v1, v2.transpose(-2, -1))
        v4 = v3.mul(0.5)
        v5 = v4.softmax(dim=-1)
        v6 = self.dropout(v5)
        v7 = torch.matmul(v6, v2)
        return v7

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randint(0, 32, size=(2, 512, 2))
x2 = torch.randint(0, 32, size=(2, 2, 512))
