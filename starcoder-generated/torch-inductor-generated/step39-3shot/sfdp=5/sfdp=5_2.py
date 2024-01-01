
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.input = torch.nn.Linear(100, 8)
        self.pos_enc = torch.nn.Embedding(100, 8)
    def forward(self, x):
        x = self.input(x)
        pos_encode = self.pos_enc(torch.arange(x.size(1))[None]).expand(x.shape)
        output = x + pos_encode
        return output
# Inputs to the model
x1 = torch.rand((1, 31, 100))
