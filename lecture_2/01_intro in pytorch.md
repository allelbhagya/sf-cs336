- bytes_per_param = 4+4+(4+4) = 16 

- ques: how long it would take to train 70b parameter model on 15T tokens on 1024 h100s?
total_flops, mfy, flops_per_day
- days=total_flops/flops_per_day

# tensors

memory setup

- float32
- float64
- memory of a tensor = number of elements x bytes taken by each element
- float16, half precision -> dynamic range isnt great -> can get instability when training
- bfloat16 - brain floating point -> in DL we care more about the dynamic range more than fraction
- optimizer states and params float32 still better
- fp8 bf16 - can get instability
- mixed precision training, attention f32 and bf16 for ff-nn

## tensors on GPU

- when want to use gpu-> move the data from cpu to gpu memory
- pytorch tensors are pointers to meta data of matrix
- copy makes new matrix
- in general we perform operators for every example in a batch and token in a sequence
- batch,sequence
- x = torch.ones(4,8,16,32) batch, sequence, matrix x,y
- w = torch.ones(32,2)
- y = x @ w
- y = (4,8,16,2)
 
### einops 
library for manipulating tensors where dimensions are named

# flop - floating point operations 
- measure of computation done
- basic operators like add (x+y) or multiply (x,y)
- without sparsity 50% less flops
- 8 h100s for 2 weeks

```
total_flops = 8 x (60 x 60 x 24 x 7) * h100_flop_per_sec 
# inspecting total flops
```


# linear model

- n points
- each point is d dimension
- the linear model maps each d dim vector to a k outputs

```
n = 1024
d = 256
k = 64

device = get_device()
x = torch.ones(n,d) 1024 x 256
w = torch.randn(d,k) 256 x 64
y = x @ w

we have one multiplication x[i][j] x w [j][k] and one addition per (i,j,k) triple
actual_num_flops = 2 x n x d x k
actual_num_flops = 33554432

(d k) number of params
flops required for forward pass is 2 x n x params
```


## mfu - model flops utilization

```
actual flops / promised flops
mfu = actual_flops_per_sec/ promised_flop_per_se
mfu >= 0.5 is good and will be higher if matmuls dominate
```