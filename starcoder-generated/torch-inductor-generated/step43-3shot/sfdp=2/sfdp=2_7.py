
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale_factor = 1.0
 
    def forward(self, input_tensor, key, value, query, dropout_p=0.0):
        # query and key: Batch * Head * Len_qk * D_qk
        # value: Batch * Head * Len_v * D_v
        # output: Batch * Head * Len_v * D_v
        # dropout_p: dropout probability
        qk = torch.matmul(query, key.transpose(-2, -1))
        inv_scale_factor = torch.tensor([1.0 / self.scale_factor], device=qk.device)
        scaled_qk = qk.div(inv_scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p)
        output1 = dropout_qk.matmul(value)
        return output1

# Initializing the model
m = Model()

# Inputs to the model
input_tensor = torch.randn(3, 6, 7)
key = torch.rand(3, 6, 10)
value = torch.rand(3, 6, 10)
query = torch.rand(3, 6, 10)
dropout_p = 1.0
