
class Model(torch.nn.Module):
    def __init__(self, dropout_p=0.5, num_values=64, num_queries=128):
        super().__init__()
        self.query = torch.nn.Linear(20, 100)
        self.key = torch.nn.Linear(20, 100)
        self.value = torch.nn.Linear(20, 20)
        self.num_queries = num_queries
        self.num_values = num_values
        self.dropout_p = dropout_p
        self.inv_scale_factor = np.sqrt(num_values)
 
    def forward(self, x1, x2):
        q = self.query(x1)
        k = self.key(x2)
        v = self.value(x2)
        qk = torch.matmul(q, k.transpose(-2, -1))
        scaled_qk = qk.div(self.inv_scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, self.dropout_p)
        output = dropout_qk.matmul(v)
        return output

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1000, 20)
x2 = torch.randn(10000, 20)
