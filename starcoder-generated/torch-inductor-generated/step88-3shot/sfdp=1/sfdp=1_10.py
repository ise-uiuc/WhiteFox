
class Model(torch.nn.Module):
    def __init__(self, query_num, key_num, value_num, inv_scale_factor, dropout_p):
        super().__init__()
        self.query = torch.nn.Linear(query_num, key_num, bias=False)
        self.key = torch.nn.Linear(key_num, key_num, bias=False)
        self.value = torch.nn.Linear(value_num, value_num, bias=False)
        if inv_scale_factor is None:
            self.inv_scale_factor = 1.0
        else:
            self.inv_scale_factor = inv_scale_factor
        self.softmax = torch.nn.Softmax(dim=-1)
        self.dropout = torch.nn.Dropout(p=dropout_p)
 
    def forward(self, xq, xk, xv):
        qk = torch.matmul(xq, xk.transpose(-2, -1))
        scaled_qk = qk / self.inv_scale_factor
        softmax_qk = self.softmax(scaled_qk)
        dropout_qk = self.dropout(softmax_qk)
        output = torch.matmul(dropout_qk, xv)
        return output

# Initializing the model
model = Model(query_num=3, key_num=3, value_num=3, inv_scale_factor=3.0, dropout_p=0.1)

# Inputs to the model
xq = torch.randn(128, 3)
xk = torch.randn(256, 3)
xv = torch.randn(256, 3)
