
class Model(torch.nn.Module):

    def __init__(self, num_heads):
        super().__init__()
        self.num_heads = num_heads

        self.dropout_p = 0.0
        self.inv_scale_factor = 1.0 / math.sqrt(self.num_heads)

        self.dropout1 = torch.nn.Dropout(p=self.dropout_p)
        self.dropout2 = torch.nn.Dropout(p=self.dropout_p)

        self.dot_product_weights1 = torch.nn.Parameter(torch.ones(self.num_heads))
        self.dot_product_weights2 = torch.nn.Parameter(torch.ones(1))

    def forward(self, query, key, value):
        scaled_dot_product1 = torch.matmul(query, key.transpose(-2, -1)) * self.inv_scale_factor
        softmax_qk1 = scaled_dot_product1.softmax(dim=-1)
        dropout_qk1 = self.dropout1(softmax_qk1)
        output1 = torch.matmul(self.dropout2(softmax_qk1), value)

        dropout_qk2 = output1 * self.dot_product_weights1.unsqueeze(1)
        output2 = dropout_qk2 * self.dot_product_weights2

        return output1, output2

# Initializing the model
m = Model(num_heads=2)

# Inputs to the model
query = torch.randn(1, 2, 8, 3)
key = torch.randn(1, 2, 2, 4)
value = torch.randn(1, 2, 2, 4)
__qkv_output__, __mult_output__ = m(query, key, value)

