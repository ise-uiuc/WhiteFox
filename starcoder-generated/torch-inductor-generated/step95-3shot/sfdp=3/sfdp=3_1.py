
class SelfAttention(nn.Module):
    def __init__(self, input_dim, num_heads, dropout_p):
        super().__init__()

        self.input_dim=input_dim   
        self.num_heads = num_heads
        self.dropout_p = dropout_p

        self.Wq = nn.Linear(input_dim, input_dim)
        self.Wk = nn.Linear(input_dim, input_dim)
        self.Wv = nn.Linear(input_dim, input_dim)

    def scaled_dot_product_attention(self, q, k, v, mask=None):
        wk = torch.matmul(q, k.transpose(-2, -1))
        scaled_qk = wk.mul(1/(self.input_dim)**0.5)

