
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout_p = 0.3
 
    def forward(self, q, k, v, inv_scf):
        dot_product = torch.matmul(q, k.transpose(-2, -1))
        scaled_product = dot_product.div(inv_scf)
        softmax_product = scaled_product.softmax(dim=-1)
        dropout_product = torch.nn.functional.dropout(softmax_product, p=self.dropout_p)
        output = dropout_product.matmul(v)
        return output

# Initializing the model
m = Model()

# Inputs to the model
q = torch.randn(1, 64, 64)
k = torch.randn(1, 64, 64)
v = torch.randn(1, 64, 64)
inv_scf = 1.0 / k.shape[-1]**0.5
