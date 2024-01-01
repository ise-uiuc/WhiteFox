
class Model(torch.nn.Module):
    def __init__(self, num_heads, embedding):
        super().__init__()
        self.num_heads = num_heads
        self.embedding = embedding
        self.dropout_p = 0.1
 
    def forward(self, x):
        v1 = x.transpose(0, 1)
        q = v1[:, 0:1, :, :]
        k = v1[:, 1:2, :, :]
        v = v1[:, 2:3, :, :]
 
        num_h = self.num_heads
        q1 = q.transpose(-2, -1)
        k1 = k.transpose(-2, -1)
        scaled_qk = q1.matmul(k1)
        inv_scale_factor = self.embedding ** 0.5
        softmax_qk = scaled_qk.div(inv_scale_factor)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=self.dropout_p)
        output = dropout_qk.matmul(v)
        output = output.transpose(0, 1).transpose(-1, -2)
        return output
        
# Initializing the model
m = Model(num_heads=8, embedding=128)

# Inputs to the model
x = torch.randn(96, 3, 64, 64)
