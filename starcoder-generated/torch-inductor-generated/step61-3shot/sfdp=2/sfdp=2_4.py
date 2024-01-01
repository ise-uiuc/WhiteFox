
class SelfAttention(torch.nn.Module):
    def __init__(self, num_heads, input_size, inner_size, dropout):
        self.dropout = dropout
        super().__init__()
        self.query_mapping = Linear(input_size, inner_size, bias=False)
        self.key_mapping = Linear(input_size, inner_size, bias=False)
        self.value_mapping = Linear(input_size, inner_size, bias=False)
        self.output_mapping = Linear(inner_size, input_size)
 
    def forward(self, x1):
        qk = torch.matmul(self.query_mapping(x1), self.key_mapping(x1).transpose(-2, -1))
        scaled_qk = qk.div(torch.sqrt(torch.tensor(float(inner_size)).float()))
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=self.dropout)
        output = self.output_mapping.forward(torch.matmul(dropout_qk, self.value_mapping(x1)))
        return output

# Initializing the model
m = SelfAttention(16, 32, 64, 0.1)

# Inputs to the model
x1 = torch.randn(1, 16, 32)
