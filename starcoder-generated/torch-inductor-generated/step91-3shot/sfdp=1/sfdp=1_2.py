
class BertAttModel(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_attention_heads = config.attention_heads
        self.hidden_size = config.hidden_size

        self.matmul1 = torch.nn.Linear(self.hidden_size, self.num_attention_heads * self.hidden_size)
        self.matmul2 = torch.nn.Linear(self.hidden_size, self.num_attention_heads * self.hidden_size, bias=False)
        self.matmul3 = torch.nn.Linear(self.hidden_size, self.num_attention_heads * self.hidden_size, bias=False)
    
    def forward(self, query, key, value, dropout_p):
        inv_scale_factor = torch.rsqrt(torch.Tensor([self.num_attention_heads*self.hidden_size*128])).to(query.device)

        qk = self.matmul1(query)
        qk = torch.matmul(qk, key.transpose(-2,-1)/inv_scale_factor.unsqueeze(0).unsqueeze(0))
        
        dropout_qk = torch.nn.functional.dropout(qk, p=dropout_p)
        softmax_qk = torch.nn.Softmax(dim=-1)(dropout_qk)
        output = torch.matmul(softmax_qk, value)

        return output

# Initializing the query/key/value tensors
query = torch.randn(1, 32, 4)
key = torch.randn(1, 32, 4)
__value = value = torch.randn(1, 32, 4)

# Dropout probability
dropout_p = 0.1

# Initializing the model
model = BertAttModel.from_pase(model_name, dropout_p=dropout_p)

print(__value)
# Inputs to the model
output = model(query, key, value, dropout_p)

