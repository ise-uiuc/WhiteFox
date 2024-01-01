
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.query, self.key, self.value = torch.nn.Parameter(torch.randn(10, 3, 5, 5)), \
                                           torch.nn.Parameter(torch.randn(10, 8, 5, 5)), \
                                           torch.nn.Parameter(torch.randn(10, 8, 5, 5))
        self.inv_scale_factor = torch.nn.Parameter(torch.randn(10, 1))
        self.dropout_p = torch.nn.Parameter(torch.randn(10, 1))
 
    def forward(self, queries):
        qk = torch.matmul(queries, self.key.transpose(-2, -1))
        scaled_qk = qk.div(self.inv_scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=self.dropout_p)
        output = dropout_qk.matmul(self.value)
        return output

# Initializing the model
m = Model()

# Inputs to the model
queries = torch.randn(1, 10, 3, 5, 5)
