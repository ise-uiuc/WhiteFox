
class Model(torch.nn.Module):
    def __init__(self,
                 input_dim:int,
                 num_heads:int):
        super().__init__()
        self.key = torch.nn.Linear(input_dim, input_dim)
        self.query = torch.nn.Linear(input_dim, input_dim)
        self.value = torch.nn.Linear(input_dim, input_dim)
        self.inv_scale_factor = num_heads ** -0.5
        self.dropout_p = 0.5
 
    def forward(self,
                x1):
        k = self.key(x1)
        q = self.query(x1)
        v = self.value(x1)
        qk = torch.matmul(q, k.T)
        scaled_qk = qk.div(self.inv_scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk,
                                                  p=self.dropout_p)
        output = dropout_qk.matmul(v)
        return output

# Initializing the model
m = Model(input_dim=16,
          num_heads=8)

# Inputs to the model
x1 = torch.randn(1, 16)
x2 = torch.randn(1, 16)
x3 = torch.randn(1, 16)
