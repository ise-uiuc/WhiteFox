
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout_p = params[0]
        self.scale_factor = math.sqrt(1.0 / params[1])
 
    def forward(self, q, k, v):
        qk = torch.matmul(q, k.transpose(-2, -1))
        scaled_qk = qk.div(self.scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=self.dropout_p)
        output = dropout_qk.matmul(v)
        return output

# Initializing the model
dropout_p = 0.9
num_heads = 2
d_model = 4
d_k = d_model // num_heads
d_v = d_model // num_heads
m = Model()

# Inputs to the model
query = torch.randn(1, 2, 4, d_model)
key = torch.randn(1, 3, 4, d_model)
value = torch.randn(1, 3, 4, d_model)
