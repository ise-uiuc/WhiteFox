
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc_qk = torch.nn.Linear(dim_hidden, dim_hidden, bias=False)
        self.fc_v  = torch.nn.Linear(dim_hidden, dim_hidden, bias=False)
        self.fc_o  = torch.nn.Linear(dim_hidden, dim_hidden, bias=False)
 
    def forward(self, input):
        q   = self.fc_qk(input)
        k   = self.fc_qk(input)
        v   = self.fc_v(input)
        qk  = torch.matmul(q, k.transpose(-2, -1))
        inv_scale_factor = math.sqrt(dim_hidden)
        scaled_qk = qk.div(inv_scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p)
        output = dropout_qk.matmul(v)
        return self.fc_o(output)

# Initializing the model
m = Model()

# Inputs to the model
x  = torch.randn(16, dim_hidden, 4)
