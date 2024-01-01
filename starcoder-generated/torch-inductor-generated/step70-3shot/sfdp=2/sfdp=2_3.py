
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout_p = torch.nn.Parameter(torch.tensor([0.0]))
 
    def forward(self, query, key, value):
        qk = torch.matmul(query, key.transpose(-2, -1))
        inv_scale_factor = torch.tensor([1.0 / query.shape[-1]**0.5])
        scaled_qk = qk.div(inv_scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=self.dropout_p)
        output = dropout_qk.matmul(value)
        return output

# Initializing the model
m = Model()

# Inputs to the model
query = torch.randn(1, 4, 64)
key = torch.randn(2, 4, 128)
value = torch.randn(2, 4, 128)
