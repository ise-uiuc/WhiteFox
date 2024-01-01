
class Attention(torch.nn.Module):
    def __init__(self, attention_hidden_dim, attention_dropout):
        super().__init__()
        self.qkv_project = lambda x: torch.nn.functional.linear(
            x, attention_hidden_dim, bias=False
        )
        self.scale_factor = 1 / math.sqrt(attention_hidden_dim / 3)
        self.dropout_p = attention_dropout
 
    def forward(self, q, k, v):
        qkv = torch.cat([self.qkv_project(q),self.qkv_project(k),self.qkv_project(v)], dim=-1)
        qk = torch.matmul(qkv, qkv.transpose(-2, -1))
        scaled_qk = qk * self.scale_factor
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(
            softmax_qk, p=self.dropout_p, training=self.training
        )
        output = torch.matmul(dropout_qk, qkv)
        return output

# Initializing the model
attention = Attention(attention_hidden_dim=128, attention_dropout=0.6)

# Inputs to the model
q = torch.randn(1, 3, 128)
k = torch.randn(4, 3, 64)
v = torch.randn(4, 3, 64)
