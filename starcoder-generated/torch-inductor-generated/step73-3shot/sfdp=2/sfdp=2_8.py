
def compute_scaled_dot_product(x1, x2, scale_factor):
    return torch.matmul(x1, x2.transpose(-2, -1)).div(scale_factor)
 
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.num_heads = 8
        self.head_size = 64 // self.num_heads
        self.input_linear = torch.nn.Linear(64, 64)
        self.output_linear = torch.nn.Linear(64, 64)
        self.query = torch.nn.Linear(64, 64)
        self.key = torch.nn.Linear(64, 64)
        self.value = torch.nn.Linear(64, 64)
        self.dropout_p = 0.0

 
    def forward(self, x1, mask):
        v1 = self.input_linear(x1)
        q = self.query(v1)
        k = self.key(v1)
        v = self.value(v1)
        x2 = compute_scaled_dot_product(q, k, 1.0 / math.sqrt(self.head_size))
        x3 = torch.nn.functional.softmax(x2, dim=-1)
        x4 = torch.nn.functional.dropout(x3, p=self.dropout_p)
        x5 = torch.matmul(x4, v)
        x4_4d = x4.permute(0, 2, 1).reshape(x5.shape)
        x5_4d = x5.permute(0, 2, 1).reshape(x4.shape)
        x6 = x4_4d * x5_4d
        x7 = x6.reshape(x1.shape)
        x8 = self.output_linear(x7)
        return x8

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 64, 64, 64)
x2 = torch.zeros(1, 64, 64, 64)
x2[:, :, [0, -1], :] = 1
