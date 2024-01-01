
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x1, x2):
        qk = x1 @ x2.transpose(-2, -1) / math.sqrt(x1.size(-1))
        qk = qk + qm # attention mask
        attn_weight = torch.softmax(qk, dim=-1) # softmax
        attn_weight = torch.dropout(attn_weight, dropout_p, True) # dropout
        output = attn_weight @ x2
        return output

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(4, 3, 64)
x2 = torch.randn(4, 5, 64)
