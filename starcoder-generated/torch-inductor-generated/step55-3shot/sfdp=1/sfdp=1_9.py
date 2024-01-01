
class Model(torch.nn.Module):
    def __init__(self, dropout_p):
        super().__init__()
        self.dropout_p = dropout_p
 
    def forward(self, k, v, q):
        qk = torch.matmul(q, k.transpose(-2, -1))
        inv_scale_factor = (torch.tensor(self.layer_depth, dtype=torch.float32).numpy() ** -0.5).item()
        scaled_qk = qk.div(inv_scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=self.dropout_p)
        output = dropout_qk.matmul(v)
        return output

# Initializing the model
m = Model(dropout_p=0.5)

# Inputs to the model
k = torch.randint(10, (2, 4, 16), dtype=torch.float32)
v = torch.randn(2, 7, 16)
q = torch.randn(3, 4, 16)
