
class Model(torch.nn.Module):
    def __init__(self, num_heads=1, dropout_p=0.0):
        super().__init__()
        self.num_heads = num_heads
        self.scale_factor = 1 / math.sqrt(num_heads)
        self.dropout_p = dropout_p
        self.W_q = torch.nn.Linear(512, 512)
        self.W_k = torch.nn.Linear(512, 512)
        self.W_v = torch.nn.Linear(512, 512)
        self.fc = torch.nn.Linear(512, 512)
 
    def forward(self, x1, x2):
        q = self.W_q(x1)
        k = self.W_k(x2)
        v = self.W_v(x2)
        q_k = torch.matmul(q.unsqueeze(-2), k.transpose(-2, -1).unsqueeze(1)).view(-1, self.num_heads, q.size(-2), k.size(-1))
        q_k = q_k * self.scale_factor
        softmax_q_k = torch.nn.functional.softmax(q_k, dim=-1)
        dropout_q_k = torch.nn.functional.dropout(softmax_q_k, p=self.dropout_p)
        output = torch.matmul(dropout_q_k.unsqueeze(-2), v).view(-1, q.size(-2), q.size(-2))
        output = output.contiguous().view(q.size(0), q.size(1), q.size(2))
        output = self.fc(output)
        return output

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 512, 16, 64)
x2 = torch.randn(1, 512, 16, 128)
