
import math
def positional_encoding(pos_encoding, batch_size, emb_size):
    pos = torch.arange(pos_encoding, dtype=torch.float).unsqueeze(1)
    i = torch.arange(0, emb_size, 2.0)
    angle_rates = 1 / torch.pow(10000, (2 * (i // 2)) / emb_size)
    angle_rads = pos * angle_rates

    angle_rads[:, 0::2] = torch.sin(angle_rads[:, 0::2])
    angle_rads[:, 1::2] = torch.cos(angle_rads[:, 1::2])
    pos_encoding = angle_rads[None,...]
    pos_encoding = pos_encoding.repeat(batch_size, 1, 1)
    return pos_encoding


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.attn_mask = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0], [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0]])

    def attention(self, Q, K, V, attn_mask):
        attn = torch.matmul(Q, K.transpose(-2, -1))
        attn = attn / math.sqrt(Q.size(-1))
        attn = attn + attn_mask[:, None, None, :]

        attn = torch.softmax(attn, dim=-1)
        attn = torch.dropout(attn, 0.1, training=True)
        output = torch.matmul(attn, V)
        return attn, output

    def forward(self, x1):
        q = x1
        k = x1
        v = x1
        attn, output = self.attention(q, k, v, self.attn_mask)
        return output

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(2, 4, 8)
