
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, queries, keys, values, dropout=0.2, scale_factor=1/np.sqrt(512)):
        qk = torch.matmul(queries, keys.transpose(-2, -1))
        scaled_qk = qk.mul(scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout)
        return dropout_qk.matmul(values)
 
# Initializing the model
m = Model()

# Inputs to the model
queries = torch.randn(1, 50, 48, 512)
keys = torch.randn(1, 35, 512, 48)
values = torch.randn(1, 35, 48, 512)
dropout_p = 0.1
