
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def foward(self, input_tensor):
        query = input_tensor[:, :128].unsqueeze(-1)
        key = input_tensor[:, 128:256]
        value = input_tensor[:, 256:]
        inv_scale_factor = 0.0625
        dropout_p = 0
        qk = torch.matmul(query, key.transpose(-2, -1))
        scaled_qk = qk.div(inv_scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p)
        output = dropout_qk.matmul(value)
        return output

# Initializing the model
m = Model()

# Inputs to the model
input_tensor = torch.randn(1, 384)
