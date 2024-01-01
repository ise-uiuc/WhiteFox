
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = torch.nn.MultiheadAttention(32, 4, dropout=0.25, batch_first=True)

    def forward(self, tensor1):
        v1 = self.attn(tensor1)
        return v1

# Initializing the model
m = Model()

# Inputs to the model
tensor1 = torch.randn(1, 4, 32, 32)
