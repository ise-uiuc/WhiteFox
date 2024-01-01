
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.inv_scale_factor = torch.tensor([64])
        self.dropout_p = torch.tensor([0.1])
    
    def forward(self, x1, x2):
        v1 = torch.matmul(x1, x2.transpose(-2, -1))
        v2 = v1.div(self.inv_scale_factor)
        v3 = v2.softmax(dim=-1)
        v4 = torch.nn.functional.dropout(v3, self.dropout_p, True)
        v5 = v4.matmul(x2)
        return v5

# Initializing the model and inputs
m = Model()
x1 = torch.randn(1, 8, 8, 8)
x2 = torch.randn(1, 8, 8, 8)
