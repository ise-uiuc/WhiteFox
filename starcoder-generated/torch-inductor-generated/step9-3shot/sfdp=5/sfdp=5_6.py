
class SelfAttention(torch.nn.Module):
    def __init__(self, dropout):
        super().__init__()
        self.dropout = dropout
 
    def forward(self, q, k, v, scale_factor):
        t = q.size(-1)
        qk = q @ k.transpose(-2, -1) / math.sqrt(t)
        qk = qk + q.new_ones(t, t) * (v.size(-2) + v.size(-1))
        attn_weight = torch.softmax(qk, dim=-1)
        attn_weight = torch.dropout(attn_weight, self.dropout, True)
        output = attn_weight @ v
        return output

class TransformerLayer(torch.nn.Module):
    def __init__(self, q_channal_num, k_channal_num, v_channal_num, head_num, dropout, scale_factor):
        super().__init__()
        self.self_attention = SelfAttention(dropout)
        self.linear1 = torch.nn.Linear(v_channal_num, v_channal_num)
        self.linear2 = torch.nn.Linear(v_channal_num, v_channal_num)
 
    def forward(self, q, k, v):
        x = self.self_attention(q, k, v, scale_factor)
        x = self.linear1(x)
        x = torch.relu(x)
        x = torch.dropout(x, 0.1)
        x = self.linear2(x)
        x = torch.relu(x)
        x = torch.dropout(x, 0.1)
        return x

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer_layer = TransformerLayer(3, 96, 96, 4, 0.1, 1)
 
    def forward(self, x, padding_mask):
        x = self.transformer_layer(x, x, x)
        return x

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(12, 12, 96)
padding_mask = torch.eye(12, 12).bool()
