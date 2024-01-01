
class Model(torch.nn.Module):
    # The model expects two inputs: input1 and input2
    def __init__(self):
        super().__init__()
 
    def forward(self, i1, i2):
        qk = torch.matmul(i1, i2.transpose(-2, -1))
        scaled_qk = qk.div(1.0 / math.sqrt(dim_k))
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p)
        output = dropout_qk.matmul(value)
        return output

# Initializing the model
m = Model()

# Inputs to the model
i1 = torch.randn(1, seq_length, dim_model)
i2 = torch.randn(1, seq_length, dim_model)
