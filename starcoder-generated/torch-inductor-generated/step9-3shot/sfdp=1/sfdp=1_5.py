
class SelfAttention(torch.nn.Module):
    def __init__(self, dim, inv_scale_factor, dropout_p):
        super().__init__()

        self.q = torch.nn.Linear(dim, dim)
        self.k = torch.nn.Linear(dim, dim)
        self.v = torch.nn.Linear(dim, dim)
        self.out = torch.nn.Linear(dim, dim)
        self.inv_scale_factor = inv_scale_factor
        self.dropout_p = dropout_p

    def forward(self, x1):
        q = self.q(x1)
        k = self.k(x1)
        v = self.v(x1)

        qk = torch.matmul(q, k.transpose(-2, -1))
        scaled_qk = qk.div(self.inv_scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(
            softmax_qk, p=self.dropout_p
        )
        output = dropout_qk.matmul(v)

        return output

# Initializing the model
m = SelfAttention(128, inv_scale_factor=1.0, dropout_p=0.0)

# Inputs to the model
x1 = torch.randn(1, 128)
