
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, v1, v2, q1, q2, k1, k2):
        qk = (q1 @ k1.transpose(-1, -2) + q2 @ k2.transpose(-1, -2)) / math.sqrt(q1.size(-1))
        qk = qk + torch.autograd.Variable(torch.zeros(qk.shape), requires_grad=False)
        attn_weight = torch.softmax(qk, dim=-1)
        dropout_p = 0.5
        attn_weight = torch.dropout(attn_weight, dropout_p)
        output = attn_weight @ v1 + attn_weight @ v2
        return v1, v2, q1, q2, k1, k2, output
 
# Initializing the model
m = Model()

# Inputs to the model
v1 = torch.randn(2, 8, 4, 8)
v2 = torch.randn(2, 8, 4, 16)
q1 = torch.randn(2, 4, 8)
q2 = torch.randn(2, 4, 8)
k1 = torch.randn(2, 4, 16)
k2 = torch.randn(2, 4, 16)
__v1__, __v2__, __q1__, __q2__, __k1__, __k2__, 