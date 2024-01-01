
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__(dropout_p=0, attn_mask=0)
 
    def forward(self, q, k, v, **kwargs):
        qk = torch.matmul(q, k.transpose(-2, -1))
        qk = qk / math.sqrt(q.shape[-1])
        qk = qk + kwargs["attn_mask"]
        attn_weight = torch.softmax(qk, dim=-1)
        attn_weight = torch.dropout(attn_weight, self.dropout_p, True)
        output = torch.matmul(attn_weight, v)
        return output, attn_weight

# Initializing the model
m = Model()

# Inputs to the model
q = torch.randn(1, 20, 128)
k = torch.randn(1, 20, 128)
v = torch.randn(1, 20, 128)
__output1__, __output2__ = m(q, k, v, attn_mask=self.full_attn_mask(q, k))

