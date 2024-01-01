
class Model(torch.nn.Module):
    def __init__(self, d_model, nheads):
        super().__init__()
        self.nheads = nheads
        self.att_weights = torch.nn.Linear(d_model, 1)
 
    def forward(self, x1, x2):
        v1 = x1.matmul(x2.transpose(-2, -1)) / math.sqrt(x1.size(-1))
        v2 = self.att_weights(v1).softmax(-1).transpose(-1, -2)
        v3 = v2.matmul(x2).transpose(1, 2)
        return v3

# Initializing the model
m = Model(d_model=64, nheads=1)

# Inputs to the model
x1 = torch.randn(1, 3, 128, 64)
x2 = torch.randn(1, 5, 128, 64)
