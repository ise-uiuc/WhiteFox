
class Model(torch.nn.Module):
    def __init__(self, query, key, value, *, dropout, inv_scale_factor):
        super().__init__()
        if self.training:
            dropout_p = dropout
        else:
            dropout_p = 0
        dropout_qk, _ = torch.nn.functional.dropout(torch.nn.functional.softmax(query.matmul(key.transpose(-2, -1)).div(inv_scale_factor), dim=-1), p=dropout_p)
        self.output = dropout_qk.matmul(value)
 
    def forward(self, x1):
        return self.output

# Initializing the model
m = Model(x1, x2, x3)

# Inputs to the model
x1 = torch.randn(1, 4, 8)
x2 = torch.randn(10, 4, 8)
x3 = torch.randn(10, 6, 2)
