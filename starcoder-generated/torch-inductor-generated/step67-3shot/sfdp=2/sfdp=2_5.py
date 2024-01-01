
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.key = torch.nn.Parameter(torch.zeros([1, 8, 64, 64]))
        self.query = torch.nn.Parameter(torch.zeros([1, 8, 64, 64]))
        self.value = torch.nn.Parameter(torch.zeros([1, 8, 64, 64]))
        self.dropout_p = torch.nn.Parameter(torch.zeros([]))
        self.scale_factor = 8
        self.inv_scale_factor = 1 / self.scale_factor
 
    def forward(self, _):
        qk = torch.matmul(self.query, self.key.transpose(-2, -1))
        scaled_qk = qk.div(self.inv_scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=self.dropout_p)
        output = dropout_qk.matmul(self.value)
        return output

# Initializing the model
m = Model()

# Inputs to the model
_ = torch.randn([])
