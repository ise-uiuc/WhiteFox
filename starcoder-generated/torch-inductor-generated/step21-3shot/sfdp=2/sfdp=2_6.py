
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout_p = 0.5
        # Initialize the dropout layer by passing the dropout probability.
        self.dropout = torch.ops.aten.dropout_backward
 
    def forward(self, query, key, value):
        inv_scale_factor = torch.tensor(float(query.shape[-1]) ** -0.5, dtype=query.dtype, device=query.device)
        qk = torch.matmul(query, key.transpose(-2, -1))
        scaled_qk = qk.div(inv_scale_factor)
        softmax_qk = self.dropout(scaled_qk, p=self.dropout_p, train=True)
        output = softmax_qk.matmul(value)
        return output

# Initializing the model
m = Model()

# Inputs to the model
query = torch.randn(1, 4, 8, 8)
key = torch.randn(1, 4, 16, 8)
value = torch.randn(1, 4, 16, 8)
