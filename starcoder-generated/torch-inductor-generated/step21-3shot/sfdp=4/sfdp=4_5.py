
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, x1):
        qk = x1 @ x1.transpose(-2, -1) / math.sqrt(x1.size(-1))
        attn_mask = -10000.0 * torch.eye(1280).to(x1.device) # A 1280x1280 diagonal matrix with each value of 10000.0
        attn_weight = torch.softmax(qk + attn_mask, dim=-1)
        output = attn_weight @ x1
        return output

# Initializing the model
m = Model()
x1 = torch.randn(1, 1280, 32)
