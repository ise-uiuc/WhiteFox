
class Model(torch.nn.Module):
    def __init__(self, attn_heads=4, attn_dropout_p=0.0, attn_softmax_p=0.5, proj_dropout_p=0.0):
        super().__init__()
        self.scale_factor = (attn_heads * attn_heads * 3) ** -0.25
        self.dropout_p = attn_dropout_p
 
    def forward(self, x1, x2):
        v1 = torch.matmul(x1, x2.transpose(-2, -1))
        v2 = v1.div(self.scale_factor)
        v3 = v2.softmax(dim=-1)
        v4 = F.dropout(v3, p=self.dropout_p)
        v5 = torch.matmul(v4, x2)
        return v5

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 8, 64, 64)
x2 = torch.randn(1, 8, 64, 64)
