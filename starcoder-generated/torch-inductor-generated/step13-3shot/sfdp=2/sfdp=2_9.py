
class Model(torch.nn.Module): 
    def __init__(self, n_head, dim_model):
        super().__init__()
        self.n_head = n_head
        self.dim_model = dim_model
        self.inv_scale_factor = dim_model ** -0.5
        self.dropout_p = 0.
        self.w_query = torch.nn.Linear(dim_model, dim_model)
        self.w_key = torch.nn.Linear(dim_model, dim_model)
        self.w_value = torch.nn.Linear(dim_model, dim_model)
 
    def forward(self, x1, x2, mask):
        query = self.w_query(x1)
        key = self.w_key(x2)
        value = self.w_value(x2)
        qk = torch.matmul(query, key.transpose(-2, -1))
        scaled_qk = qk.div(self.inv_scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=self.dropout_p)
        output = dropout_qk.matmul(value)
        return output

# Initializing the model
n_head = 32
dim_model = 512
dim_feedforward = 2048
dropout_p = 0.1
m = Model(n_head, dim_model)

# Inputs to the model
x1 = torch.randn(2, 64, 512)
x2 = torch.randn(2, 4096, 512)
mask = torch.randn(2, n_head, 64, 4096)
