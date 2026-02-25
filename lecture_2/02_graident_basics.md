to compute gradient backwards


```
y = 0.5 (x * w-5) ^2

forward pass: compute loss

x = torch.tensor([1.,2,3])
w = torch.tensor([1.,1,1], required_grad = True) # want gradient
pred_y = x @ w
loss = 0.5 * (pred_y - 5).pow(2)
```

```
backward pass
loss.backward()
assert loss.grad is None
asser pred_y.grad is None
assert x.grad is None
assert torch.equal(w.grad, torch.tensor([1,2,3]))
```

## gradient flops


```
n = 1024
d = 256
k = 64

x = torch.ones(n,d)
w1 = torch.randn(d,d)
w2 = torch.randn(d,k)

model = x -> w1 -> h1-> w2 -> h2 -> loss

h1 = x @ w1
h2 = h1 @ w2
loss = h2.pow(2).mean()

```


```
forward FLOPS

- multiply x[i][j] * w1[j][k]
- add to h1[i][k]
- multiply h1[i][j] * w2[j][k]
- add to h2[i][k]

num_forward_flops = (2 x n x d x d) + (2 x n x d x k)
num_forward_flops = 2 times forward in layer 1 + 2 times forward in layer 2

```

## flops in backward

```
h1.retain_grad()
h2.retain_grad()
loss.backward()


h1.grad = d loss / d h1
h2.grad = d loss / d h2
w1.grad = d loss / d w1
w2.grad = d loss / d w2

focusing on parameter w2
```

```
total flops
forward = (2 x n x params) flops
backward = (4 x n x params) flops
total = (6 x n x params) flops
```