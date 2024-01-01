
num_heads = 10
attention_dropout_probability = 0.1

class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.qkv_project = torch.nn.Conv2d(32, num_heads, 2, stride=2)
        self.out_project = torch.nn.Conv2d(32, 32, 2, stride=2)
 
    def forward(self, x1):
        qkv = self.qkv_project(x1)
        q, k, v = torch.chunk(qkv, 3, dim=1) # Split the output of the convolution into 3 parts of a fixed size
        scaled_qk = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(q.shape[-1]) # Compute the dot product between the query and the key and scale it
        softmax_qk = scaled_qk.softmax(dim=-1) # Apply softmax to the scaled dot product
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=self.dropout_p) # Apply dropout to the softmax output
        output = dropout_qk.matmul(v) # Compute the dot product of the dropout output and the value
        return self.out_project(output)

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 32, 64, 64)
