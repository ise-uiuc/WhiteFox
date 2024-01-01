
class Model(torch.nn.Module):
    def __init__(self, *, num_queries, num_keys, num_values, dim, dropout_p):
        super().__init__()
        self.num_queries = num_queries
        self.num_keys = num_keys
        self.num_values = num_values
        self.dim = dim
        self.dropout_p = dropout_p
 
    def forward(self, inputs):
        b, nq, dim = inputs.shape
        qk = torch.matmul(inputs.reshape(b, nq, 1, dim), torch.transpose(inputs.reshape(b, 1, nq, dim), -2, -1))
        inv_scale_factor = math.sqrt(1.0 / math.pow(dim, 0.5))
        dropout_p = self.dropout_p
        softmax_qk = torch.nn.functional.dropout(torch.nn.Softmax(dim=-1)(qk.div(inv_scale_factor)), p=dropout_p)
        output = torch.matmul(softmax_qk, inputs.reshape(b, nq, dim, 1)).reshape(b, nq, dim)
        return output

# Initializing the model
m = Model(num_queries=3, num_keys=3, num_values=3, dim=4, dropout_p=0.5)

# Inputs to the model
inputs = torch.randn(4, 3, 4)
