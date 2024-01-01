
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.q_linear = torch.nn.Linear(8, 8)
        self.k_linear = torch.nn.Linear(8, 8)
        self.k_t = None
 
    def forward(self, query, value, inv_scale_factor, dropout_p):
        query = self.q_linear(query)
        key = self.k_linear(value) if self.k_t is None else self.k_t(value)
        res = query.matmul(key.transpose(-2, -1))
        scaled_res = res.div(inv_scale_factor)
        softmax_res = torch.nn.functional.softmax(scaled_res, dim=-1)
        dropout_res = torch.nn.functional.dropout(softmax_res, p=dropout_p)
        output = dropout_res.matmul(value)
        return output

# Initializing the model
m = Model()

# Inputs to the model
query = torch.randn(2, 8)
value = torch.randn(2, 8, 16)
inv_scale_factor = 2.44
dropout_p = 0.22567
