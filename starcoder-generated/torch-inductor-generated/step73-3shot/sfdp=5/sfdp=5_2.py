
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.heads = 1
        self.seq_len = 1
        self.dim = 512 // self.heads
    def forward(self, query, key, value, attn_mask):
        qk = query @ key.transpose(-2, -1) / math.sqrt(query.size(-1))
        qk = qk + attn_mask
        attn_weight = torch.softmax(qk, dim=-1)
        attn_weight = torch.dropout(attn_weight, 0.1, True)
        output = attn_weight @ value
        return output
# Inputs to the model
query = torch.tensor([[[[-0.9744, -1.3997,  0.8572,  0.3755,  0.1306],[ 0.1726,  1.4983, -0.1577, -0.8755, -0.6887],[ 0.5395,  0.8256, -0.0871, -0.7376, -1.9795]]]])
key = torch.tensor([[[[ 0.8584, -0.4259,  0.5770, -0.7509, -0.4929],[ 0.9989,  1.9660,  1.4278,  0.4952,  0.1241],[-0.5913, -0.5082,  0.1153, -0.2758, -0.0943]]]])
value = torch.tensor([[[[-0.9732, -0.4180, -1.3261,  1.1234, -1.3701],[-0.0318,  0.2494, -0.3864,  0.8821, -0.4463],[ 1.1925, -1.4421,  1.0360, -1.7741, -2.2381]]]])
attn_mask = torch.tensor([[[[ True,  True,  True, False, False]]]])
