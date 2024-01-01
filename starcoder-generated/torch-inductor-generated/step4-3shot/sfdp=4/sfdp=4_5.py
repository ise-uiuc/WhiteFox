
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x1, x2):
        qk = x1 @ x2.transpose(1, 2)
        attn_mask = (torch.triu(x2[:-1,:-1]) + torch.tril(x2[:-1,:-1])) * -10000.0
        attn_weight = torch.softmax(qk, dim=-1)
        output = attn_weight @ x2
        return None, None

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(2, 10, 3)
x2 = torch.randn(2, 10, 4)
x3 = torch.randn(2, 10, 4)
x4 = torch.randn(2, 10, 4)
