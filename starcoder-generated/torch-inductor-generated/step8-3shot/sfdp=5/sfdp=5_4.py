
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout = torch.nn.Dropout(0.2)
 
    def forward(self, x1):
        v1 = x1 @ x1.transpose(0, 1)
        v2 = v1 / math.sqrt(x1.size(1))
        v3 = v2 + __attn_mask__.to(x1.device)
        v4 = torch.softmax(v3, dim=-1)
        v5 = self.dropout(v4)
        v6 = v5 @ x1
        return v6

# Inputs to the model
x1 = torch.randn(512, 512)
