
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, query, key, value):
        qk = torch.matmul(query, key.transpose(-2, -1))
        scaled_qk = qk.div(inv_scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p) 
        return dropout_qk.matmul(value)

# Initializing the model
m = Model()

# Inputs to the model
query = torch.randn(seq_length, batch_size, hidden_size)
key = torch.randn(seq_length, batch_size, hidden_size)
value = torch.randn(seq_length, batch_size, hidden_size)
output = m(query, key, value)

# Shape of output
__output_shape__ = output.shape

# # Sample code for model_info
# import torch
# class Model(torch.nn.Module):
#     def __init__(self):
#         super().__init__()
#     def forward(self, x1):
#         x1 = torch.mean(x1, dim=(0, 1, 2))
#         return x1
# model = Model()
# model_info(model, (1, 3, 64, 64), batch_dim=0)
