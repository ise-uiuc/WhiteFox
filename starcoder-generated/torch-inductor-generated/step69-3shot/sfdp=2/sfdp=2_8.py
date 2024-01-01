
class Model(torch.nn.Module):
    def __init__(self,
                query: torch.Tensor,
                key: torch.Tensor,
                value: torch.Tensor):
        super().__init__()
        self.query = torch.nn.Parameter(query, requires_grad=False)
        self.key = torch.nn.Parameter(key, requires_grad=False)
        self.value = torch.nn.Parameter(value, requires_grad=False)
 
    def forward(self, query, key, value):
        qk = torch.matmul(query, self.key.transpose(-2, -1))
        scaled_qk = qk.div(1./64)
        softmax_qk = torch.nn.functional.softmax(scaled_qk, dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=0.3)
        output = torch.matmul(dropout_qk, self.value)
        return output

# Initializing the model
query = torch.rand([4, 64, 64])
key = torch.rand([4, 16, 256])
value = torch.rand([4, 16, 256])
m = Model(query, key, value)

# Inputs to the model
