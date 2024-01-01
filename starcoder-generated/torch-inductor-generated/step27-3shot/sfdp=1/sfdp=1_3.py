
class Model(torch.nn.Module):
    def __init__(self, dropout=0.1, ntoken=30000, ninp=768):
        super().__init__()
        self.drop = torch.nn.Dropout(dropout)
        self.embedding = torch.nn.Embedding(ntoken, ninp)
        self.softmax = torch.nn.Softmax(dim=-1)
        self.matmul = torch.matmul
 
    def forward(self, query, key, value, scale_factor):
        qk = self.matmul(query, key.transpose(-2, -1))
        scaled_qk = qk.div(scale_factor)
        softmax_qk = self.softmax(scaled_qk)
        dropout_qk = self.drop(softmax_qk)
        output = self.matmul(dropout_qk, value)
        return output

# Initializing the model
m = Model(dropout=0.4, ntoken=30000, ninp=768)

# Inputs to the model
query = torch.randn(10, 3, 768)
key = torch.randn(10, 3, 768)
value = torch.randn(10, 3, 768)
scale_factor = torch.randn(10)
