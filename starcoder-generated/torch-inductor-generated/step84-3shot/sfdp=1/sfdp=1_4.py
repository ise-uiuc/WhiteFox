
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.drop = torch.nn.Dropout(config.attention_probs_dropout_prob)
 
    def forward(self, query, key, value, dropout_p=0.0):
        scale_factor = 1 / math.sqrt(self.key.size(-1))
        qk = torch.matmul(query, key.transpose(-2, -1))
        scaled_qk = qk.div(scale_factor)
        softmax_qk = F.softmax(scaled_qk, dim=-1)
        dropout_qk = self.drop(softmax_qk)
        output = dropout_qk.matmul(value)
        return output
 
# Initializing the model
m = Model()

# Inputs to the model
query = torch.randn(1, 512, 64)
key = torch.randn(1, 512, 64)
value = torch.randn(1, 512, 64)
