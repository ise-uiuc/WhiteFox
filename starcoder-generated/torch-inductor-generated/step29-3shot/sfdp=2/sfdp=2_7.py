
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.inv_scale_factor = 2.5
 
    def forward(self, x1, x2):
        v1 = x1.shape[-1]
        v2 = x2.shape[1]
        v3 = torch.matmul(x1, x2.transpose(-2, -1))
        v4 = v3.div(self.inv_scale_factor).softmax(dim=-1)
        v5 = torch.nn.functional.dropout(v4, p=0.1)
        v6 = torch.matmul(v5, x2).permute(0, 2, 1)
        return v6

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3, 8)
x2 = torch.randn(1, 4, 8)
