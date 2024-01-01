
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2):
        x3 = x1 * x1
        x4 = torch.randn((20, 512))
        x5 = x2 * x2 
        x6 = F.dropout(x4, p=0.1, training=self.training)
        x7 = torch.randn((20, 512))
        x8 = torch.rand_like(x7)
        x_ret = torch.add(x6, x8)
        x_ret = torch.nn.functional.dropout(x1, p=0.8, training=self.training)
        y = x3 - x5
        y1 = F.dropout(y, p=0.8)
        y2 = x3 * y2
        return y2
# Inputs to the model
x1 = torch.randn(8, 32, embed_dim)
x2 = torch.randn(8, 32, embed_dim)
