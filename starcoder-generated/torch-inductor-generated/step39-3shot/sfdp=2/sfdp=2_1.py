
class Model(torch.nn.Module):
    def __init__(self, num_heads=1):
        super().__init__()
        self.multiheaded_attention = torch.nn.MultiheadAttention(embed_dim=256, num_heads=num_heads, dropout=0.1)
 
    def forward(self, x1, x2):
        x1 = x1.permute((1, 0, 2))
        x2 = x2.permute((1, 0, 2))
        qk_v = self.multiheaded_attention(x1, x1, x1)
        softmax_qk = torch.nn.functional.softmax(qk_v[0], dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=0.1)
        scaled_qk = torch.tensor(1.0 / math.sqrt(qk_v[0].size(-1))) * dropout_qk
        output = (torch.matmul(dropout_qk, qk_v[0]),)
        return output

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(3, 5, 256)
x2 = torch.randn(3, 5, 256)
