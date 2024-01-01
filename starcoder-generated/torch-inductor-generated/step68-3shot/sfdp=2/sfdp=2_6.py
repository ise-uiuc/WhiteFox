
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, x1, x2):
        q = x1 # First query
        k = x2 # First key
        v = x1 # First value
        kkv = torch.matmul(k, k.transpose(-2, -1)) # Compute the dot product of the first key and the first key
        qkv = torch.matmul(q, kkv) # Compute the dot product of the first query and the dot product of the first key and the first key
        scale = 768 ** -0.5 # Inverse scale factor
        output = torch.matmul(qkv.softmax(dim=-1), v).mul(scale) # Compute the dot product of the softmax output of the dot product of the first query and the dot product of the first key and the first key and the first value
        output = torch.nn.functional.dropout(output, p=0.1)
        return output

# Initializing the model
m = Model()

# Input tensors to the model
x1 = torch.randn(1, 64, 768)
x2 = torch.randn(1, 64, 768)
