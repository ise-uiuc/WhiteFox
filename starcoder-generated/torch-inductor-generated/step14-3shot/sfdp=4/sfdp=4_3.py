
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input1, input2):
        q = input1
        k = input2
        v = torch.randn(1, 3, 3, 3)
        qk = torch.matmul(q, k.transpose(-2, -1))
        mask = torch.randint(0, 1, size=(1, 3))
        qk = qk + mask
        attn_weight = torch.softmax(qk, dim=-1)
        output = torch.matmul(torch.matmul(v, attn_weight), -1)
        return output

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3, 3, 3)
x2 = torch.randn(1, 3, 6, 6)
