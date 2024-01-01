
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
	self.dropout_p = 0.45
 
    def forward(self, query, key, value, scale_factor, inv_scale_factor):
        qk = torch.matmul(query, key.transpose(-2, -1))
        scaled_qk = qk.div(inv_scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=self.dropout_p)
        output = dropout_qk.matmul(value)
        return output

# Initializing the model
m = Model()
query = torch.randn(2, 5, 4)
key = torch.randn(2, 3, 4)
value = torch.randn(2, 3, 6)
scale_factor = torch.rand(2, 1)
inv_scale_factor = 1.0 / (scale_factor + 1e-5)
