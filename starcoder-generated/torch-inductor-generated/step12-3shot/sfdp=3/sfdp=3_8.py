
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, x, y, attn_mask, dropout):
        qk = torch.matmul(x, y.transpose(-2, -1))
        scale_factor = attn_mask.to(torch.float) / (y.size(-2) ** 0.5)
        scaled_qk = qk.mul(scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout)
        output = dropout_qk.matmul(y)
        return output.add(x)

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(1, 5, 1)
y = torch.randn(1, 2, 1)
scale_factor = torch.randn(1, 5, 2).clamp(min=1, max=1.5)
attn_mask = torch.randn(1, 5, 2)
dropout = 0
