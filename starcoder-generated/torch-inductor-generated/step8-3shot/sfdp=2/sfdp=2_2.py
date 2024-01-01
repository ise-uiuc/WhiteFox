
class Model(torch.nn.Module):
    def __init__(self, batch_size, num_heads, sequence_length, head_size, mask_size):
        super().__init__()
        self.query = torch.nn.Parameter(torch.rand(batch_size, num_heads, sequence_length, head_size))
        self.key = torch.nn.Parameter(torch.rand(batch_size, num_heads, sequence_length, head_size))
        self.value = torch.nn.Parameter(torch.rand(batch_size, num_heads, sequence_length, head_size))
        self.softmax = torch.nn.Softmax(dim=-1)
        self.dropout = torch.nn.Dropout(p=0.0)
        self.mask = torch.tril(torch.ones((mask_size, mask_size))).float()
 
    def forward(self, x1, x2):
        qk = torch.matmul(x1, x2.transpose(-2, -1))
        scaled_qk = qk.div(self.mask.to(device=qk.device, dtype=torch.float32) + self.query.shape[-1]**(-0.25))
        softmax_qk = self.softmax(scaled_qk)
        dropout_qk = self.dropout(softmax_qk)
        output = dropout_qk.matmul(self.value)
        return output
 
# Initializing the model
m = Model(batch_size=2,
          num_heads=2,
          sequence_length=8,
          head_size=4,
          mask_size=2)
# Inputs to the model
x1 = torch.randn(m.query.shape)
x2 = torch.randn(m.key.shape)
