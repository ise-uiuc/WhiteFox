
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(4, 4, bias=False)
 
    def forward(self, x1, x2):
        qk = (x1 @ x2.transpose(-2, -1)) / math.sqrt(x1.size(-1))
        qk = qk + self.mask
        attn_weight = torch.softmax(qk, dim=-1)
        output = attn_weight @ x1
        return output

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 4, 2)
x2 = torch.randn(1, 2, 4)
x3 = torch.randn(1, 1, 1, 4)
m.mask = x3
