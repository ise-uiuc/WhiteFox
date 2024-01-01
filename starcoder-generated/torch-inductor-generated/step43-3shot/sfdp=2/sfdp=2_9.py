
class ComputeAttention(nn.Module):
    def __init__(self, dropout):
        super().__init__()
        self.dropout = torch.nn.Dropout(dropout)
        self.softmax1 = torch.nn.Softmax(dim=-1)
 
    def forward(self, query, key, value, scale):
        # query [n b h d]
        # key [n b h d]
        # value [n b w d]
        # scale [1]
        qk = torch.matmul(query, key.transpose(-2, -1))
        # qk [n b h w]
        scaled_qk = qk.div(scale)
        # scaled_qk [n b h w]
        softmax_qk = self.softmax1(scaled_qk)
        # softmax_qk [n b h w]
        dropout_qk = self.dropout(softmax_qk)
        # dropout_qk [n b h w]
        output = torch.matmul(dropout_qk, value)
        # output [n b h w d]
        return output, dropout_qk

class Model(nn.Module):
    def __init__(self, input_dim, output_dim, dropout=0.2):
        super().__init__()
        self.conv1 = nn.Conv1d(input_dim, 1000, 1)
        self.preact_layernorm = nn.LayerNorm(1000, eps=layer_norm_eps)
        self.self_attention = ComputeAttention(dropout)
        self.conv2 = nn.Conv1d(1000, output_dim, 1)
 
    def forward(self, x):
        h1 = self.conv1(x)
        h1_preact = self.preact_layernorm(h1)
        h2, _ = self.self_attention(h1_preact, h1_preact, h1, scale=1000 ** -1 / 2.0)
        out = self.conv2(h2)
        return out

# Initializing the model
m = Model(7, 5).cuda()

# Inputs to the model
x = torch.randn(13, 14, 7).cuda()
