
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout_qk = torch.nn.Dropout(p=dropout_p)
 
    def forward(self, query, key, value):
        x0 = torch.matmul(query, key.transpose(-2, -1))
        x1 = x0.div(inv_scale_factor)
        x2 = F.softmax(x1, dim=-1)
        x3 = self.dropout_qk(x2)
        x4 = torch.matmul(x3, value)
        return x4

# Initializing the model
m = Model()

# Inputs to the model
query = torch.randn(1, 3, 64, 64).requires_grad_()
key = torch.randn(1, 3, 64, 64)
value = torch.randn(1, 3, 64, 64)
