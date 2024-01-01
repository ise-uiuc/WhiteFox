
class Attention(torch.nn.Module):
    def __init__(self, num_heads, dim_embed, p=0.0):
        super().__init__()
        self.num_heads = num_heads
        self.dim_embed = dim_embed
        self.p = p

        self.softmax = torch.nn.Softmax(dim=-1)
        self.dropout = torch.nn.Dropout(p=self.p, inplace=True)

    def forward(self, q1, k1, v1):
        q1 = q1.unsqueeze(-2)
        k1 = k1.unsqueeze(-3)
        v1 = v1.unsqueeze(-3)

        qk = torch.matmul(q1, k1.transpose(-2, -1))

        scale_factor = 1 / (sqrt(self.dim_embed) * self.num_heads)
        scaled_qk = qk.mul(scale_factor)
        softmax_qk = self.softmax(scaled_qk)
        dropout_qk = self.dropout(softmax_qk)
        output = dropout_qk.matmul(v1)
        return output.squeeze(-3)

# Initializing the model
a = Attention(2, 3)

# Inputs to the model
q1 = torch.randn(1, 4, 3)
k1 = torch.randn(1, 3, 3)
v1 = torch.randn(1, 3, 3)
