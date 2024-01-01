
class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]
        x, self.attn = attention(query, key, value, mask=mask,
                                 dropout=self.dropout)
        x = x.transpose(1, 2).contiguous() \
           .view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)

# Initializing a layer
mha = MultiHeadedAttention(10, 10)

# Inputs to the model
mha.forward(torch.randn(23, 18, 10), torch.randn(23, 15, 42), torch.randn(23, 11, 78))

# Description of requirements
The model should contain the following pattern:
```
s1 = torch.reshape(inputs, shape = (batch, 1, height, width)) # Reshape the model from the batch dimension into height and width dimension
s2 = nn.functional.leaky_relu(s1, 0.2) # Apply the leaky ReLU function on the output of the reshape function
s3 = torch.mean(s2, axis = (-2, -1)) # Compute the mean value of the output of the reshape function in the height and width dimension
s4 = nn.functional.leaky_relu(s3, 0.2) # Apply the leaky ReLU function on the output of the mean value computation
s5 = s4.view(batch, 1, -1) # Reshape the model from the mean value dimension into batch dimension
s6 = torch.reshape(s5, shape = (batch, -1)) # Reshape the model from the mean value dimension to the new shape
```
This pattern aims to process data on the height and width dimension, and the new shape of the model could be different from the previous shape of the model.

# Model
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.reshape = TensorReshape()
        self.relu = nn.LeakyReLU(0.2)
        self.mean = nn.AvgPool2d(kernel_size = 11, stride=11)
        self.flatten = nn.Flatten()

    def forward(self, inputs):
        out = self.reshape(inputs, shape=(-1, 1, 7, 7))
        out = self.relu(out)
        out = self.mean(out)
        out = self.relu(out)
        out = self.flatten(out)
        return out

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(2, 3, 32, 32)
