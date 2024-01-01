
class Model(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.query = torch.nn.Linear(input_dim, input_dim, bias=False)
        self.key = torch.nn.Linear(input_dim, input_dim, bias=False)
        self.value = torch.nn.Linear(input_dim, output_dim, bias=False)
        self.scale_factor = torch.randn(1, 1, 1).abs()[0]
        self.dropout_p = torch.nn.Parameter(0 * torch.ones(1))
 
    def forward(self, x1):
      query = self.query(x1)
      key = self.key(x1)
      value = self.value(x1)
      qk = torch.matmul(query, key)
      scaled_qk = qk.div(self.scale_factor)
      softmax_qk = scaled_qk.softmax(dim=-1)
      dropout_qk = torch.nn.functional.dropout(softmax_qk, p=self.dropout_p)
      return dropout_qk.matmul(value)

# Initializing the model
m = Model(32, 10)

# Inputs to the model
x1 = torch.randn(32, 32)
