
class Model(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.w_q = torch.nn.Linear(QUERY_LEN, KEY_LEN)
 
    def forward(self):
        qk = self.w_q(query_input)
        scaled_qk = qk.mul(scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p)
        output = dropout_qk.matmul(value)
        return output

# Initializing the model
m = Model()

# Inputs to the model
query_input = torch.randn(BATCH_SIZE, QUERY_LEN)
value = torch.randn(BATCH_SIZE, VALUE_LEN)
