
class Model(torch.nn.Module):
    def __init__(self, dropout_p=0.8, scale_factor=128**-0.5):
        super().__init__()
        self.scale_factor = scale_factor
        self.dropout_p = dropout_p

    def forward(self, query, key, value, attention_mask):
        qk = torch.matmul(query, key.transpose(-2, -1))
        scaled_qk = qk * self.scale_factor
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=self.dropout_p)
        output = dropout_qk.matmul(value)
        output_after_mask = output + attention_mask * -1e10   # We don't actually encourage to apply softmax to masked positions
        masked_softmax_qk = softmax_qk * attention_mask + (1 - attention_mask) * -1e10 # We don't actually encourage to apply softmax to masked positions
        return output_after_mask, masked_softmax_qk

# Initializing the model
m = Model()

# Inputs to the model
query = torch.randn(1, 3, 64, 64)
key = torch.randn(1, 3, 64, 64)
value = torch.randn(1, 3, 64, 64)
attention_mask = torch.ones(1, 3, 64, 64).bool()
__output_after_mask__, __masked_softmax_qk__ = m(query, key, value, attention_mask)

