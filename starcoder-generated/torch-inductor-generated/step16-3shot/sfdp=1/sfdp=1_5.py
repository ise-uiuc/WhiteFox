
class Model(torch.nn.Module):
    def __init__(self, num_q, num_k, num_v, dropout_p, inv_scale_factor):
        super().__init__()
        self.w = torch.nn.Linear(num_q, num_v)
        self.x = torch.nn.Linear(num_k, num_v)
        self.dropout_p = dropout_p
        self.inv_scale_factor = inv_scale_factor
 
    def forward(self, query1, key1, value1):
        q = self.w(query1)
        k = self.x(key1)
        qk = torch.matmul(q, k.transpose(-2, -1))
        scaled_qk = qk.div(self.inv_scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=self.dropout_p)
        v = value1
        output = dropout_qk.matmul(v)
        return output

# Initializing the model (without generating the input tensor)
m = Model(2, 3, 4, 0.5, 0.1)

# Inputs to the model (without generating the input tensor)
query1 = torch.randn(1, 2, 1)
key1 = torch.randn(1, 3, 1)
value1 = torch.randn(1, 4, 1)
