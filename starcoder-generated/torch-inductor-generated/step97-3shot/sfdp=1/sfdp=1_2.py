
class Model(torch.nn.Module):
    def __init__(self, num_heads, input_size, dropout_p=0):
        super().__init__()
        self.qk = torch.nn.Linear(input_size, input_size)
        self.v = torch.nn.Linear(input_size, input_size)
 
    def forward(self, query, key, value):
        qk = self.qk(query)
        v = self.v(value)
        scaled_qk = qk.bmm(v.transpose(1, 2))
        inv_scale_factor = (0.5 * self.num_heads)**-0.5
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=self.dropout_p)
        return dropout_qk.bmm(v)

# Initializing the model and inputs to the model
m = Model(num_heads=8, input_size=128, dropout_p=0.5)
query = torch.randn(1, 50, 128)
key = torch.randn(1, 20, 128)
value = torch.randn(1, 20, 128)
