
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout = torch.nn.Dropout(p=dropout_p)
 
    def forward(self, qk, inv_scale_factor, value):
        return self.dropout(qk.softmax(dim=-1).div(inv_scale_factor).matmul(value))

# Initializing the model
m = Model()

# Inputs to the model
qk_input = torch.randn(50, 23, 512)
inv_scale_factor_input = torch.tensor(0.6)
value_input = torch.randn(50, 23, 64)
