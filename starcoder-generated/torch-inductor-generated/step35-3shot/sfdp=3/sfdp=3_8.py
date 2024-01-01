
class Model(torch.nn.Module):
    def __init__(self, dim=64, h=8, dropout_p=0.2):
        super().__init__()
        self.dim = dim
        self.head = h
        self.dk = dim // h # Dimensionality per head
        self.scale_factor = self.dk ** -0.5
        self.query = torch.nn.Parameter(torch.randn(dim, dim))
        self.key = torch.nn.Parameter(torch.randn(dim, dim))
        self.value = torch.nn.Parameter(torch.randn(dim, dim))
        self.proj = torch.nn.Linear(dim * 2, dim * 2)
        self.dropout = torch.nn.Dropout(dropout_p)
 
    def forward(self, x):
        query = self.query.unsqueeze(1)
        key = self.key.transpose(-2, -1).unsqueeze(1)
        value = self.value.unsqueeze(1)
        batch = x.size(0)
 
        qk = torch.matmul(query, key)
        scaled_qk = qk * self.scale_factor
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = self.dropout(softmax_qk)
        qkv = torch.matmul(dropout_qk, value)
        x = qkv.transpose(3, 1)
        x = x.reshape(batch, -1)
        x = self.proj(x)
        x = x.reshape(batch, -1, 2, self.dim)
        x = x.transpose(3, 1)
        output = [x[:, :, 0, :], x[:, :, 1, :]]
        return output
 
model = Model(dim=64, h=8, dropout_p=0.2)

# Inputs to the model
x = torch.randn(1, 8, 64)
