
class Model(torch.nn.Module):
    def __init__(self):
        super(self).__init__()
        self.linear = torch.nn.Linear(4,4)

    def forward(self, inputs, output_size, attention_probs_dropout_prob, hidden_dropout_prob):

        hidden_size = inputs.shape[-1]

        query = self.linear(inputs)
        query = query / np.sqrt(hidden_size)

        key = self.linear(inputs)
        key = key / np.sqrt(hidden_size)

        key_t = torch.transpose(key, -2, -1)
        dot = torch.matmul(query, key_t)

        mask = (torch.ones(inputs.shape, dtype=torch.float32))
        dot = dot + mask
        
        attention_mask = torch.ones((attention_probs.size(0), attention_probs.size(1), attention_probs.size(1)), dtype=torch.float32) # (batch_size, max_seq_length)
        attention_probs = F.dropout(input=attention_mask, p=attention_probs_dropout_prob)

        attention_probs = F.softmax(attention_probs)
        attention_probs = F.dropout(input=attention_probs, p=dropout_prob)

# Initializing the model
m = Model()

# Inputs to the model
inputs = torch.randn(8, 8, 4)
output_size = (8, 12, 4)
attention_probs_dropout_prob = 0.6
hidden_dropout_prob = 0.5
