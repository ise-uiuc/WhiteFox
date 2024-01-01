
class Model(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.qkv = torch.nn.Linear(input_dim, output_dim*2)

    def forward(self, x1):
        qkv = self.qkv(x1)
        q,k,v = torch.chunk(qkv,3,2)
        q_k = torch.matmul(q,k.transpose(-2, -1))
        scaled_q_k = q_k / np.sqrt(512.)
        softmax_q_k = scaled_q_k.softmax(dim=-1)
        dropout_q_k = torch.nn.functional.dropout(softmax_q_k, p=dropout_p)
        output = torch.matmul(dropout_q_k, v)
        return output

# Initializing the model
m = Model(512, 128)
x1 = torch.randn(8,512)
