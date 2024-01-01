
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(3, 4, bias=False)
        self.linear2 = torch.nn.Linear(4, 5, bias=False)

    def forward(self, q1_p, k1_p):
        q1 = self.linear1(q1_p)
        k1 = self.linear2(k1_p)
        qk = torch.matmul(q1, k1.transpose(-2, -1))
        inv_scale_factor = float(2 / 3) * torch.rsqrt(toq(torch.einsum('b h n, b h n -> b h', q1, k1).sum(-1)))
        scaled_qk = qk.div(inv_scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        return softmax_qk

# Initializing the model
m = Model()

# Inputs to the model
q1_p = torch.randn(3, 4)
k1_p = torch.randn(5, 4)
