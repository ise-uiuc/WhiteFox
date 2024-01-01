
class Model(torch.nn.Module):
    def __init__(self, n_heads=1):
        super().__init__()
        self.dropout_p = 0.5
 
    def forward(self, x1, x2):
        m = torch.matmul(x1, x2)
        inv_scale_factor = m.shape[-1]**-0.25
        softmax_m = m.softmax(dim=-1)
        dropout_m = torch.nn.functional.dropout(softmax_m, p=self.dropout_p)
        output = dropout_m.matmul(x2)
        return output

# Initializing the model
m = Model.__new__(Model)

# Inputs to the model
x1 = torch.randn(1, 5, 6)
x2 = torch.randn(1, 6, 7)
