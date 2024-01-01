
class Model1(torch.nn.Module):
    def __init__(self, dropout_prob):
        super().__init__()
        self.dropout_p = dropout_prob
 
    def forward(self, query, key, value, inv_scale_factor):
        qk = torch.matmul(query, key.transpose(-2, -1))
        scaled_qk = qk.div(inv_scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=self.dropout_p)
        output = dropout_qk.matmul(value)
        return output

# Initializing the model
m = Model1(0.05)

# Inputs to the model
query = torch.randn(32, 3, 64)
key = torch.randn(32, 3, 72)
value = torch.randn(32, 72, 32)
inv_scale_factor = torch.tensor(8.0, requires_grad=False)
