
class Model(torch.nn.Module):
    def __init__(self, n_head, n_hid, dropout_p):
        super().__init__()
        self.n_head = n_head
        self.n_hid = n_hid
        self.query_linear = torch.nn.Linear(n_hid, n_head)
        self.key_linear = torch.nn.Linear(n_hid, n_head)
        self.value_linear = torch.nn.Linear(n_hid, n_head)
        self.dropout_p = dropout_p
 
    def forward(self, query, key, value, inv_scale_factor):
        query = self.query_linear(query).view(query.shape[0], self.n_head, query.shape[1])
        key = self.key_linear(key).view(key.shape[0], self.n_head, key.shape[1])
        value = self.value_linear(value).view(value.shape[0], self.n_head, value.shape[1])
        qk = torch.matmul(query, key.transpose(-2, -1))
        scaled_qk = qk.div(inv_scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=self.dropout_p)
        output = dropout_qk.matmul(value)
        return output.view(output.shape[0], output.shape[1])
 
# Initializing the model
device = torch.device("cpu")
n_head = 10
n_hid = 10
dropout_p = 0.5
inv_scale_factor = 1.0 / math.sqrt(n_hid)
m = Model(n_head, n_hid, dropout_p)

# Inputs to the model
query = torch.randn(1, 100, 15)
key = torch.randn(1, 100, 15)
value = torch.randn(1, 100, 30)
