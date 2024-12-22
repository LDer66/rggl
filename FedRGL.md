
## 1. Requirements

```python
torch==1.12.1
torch-cluster==1.6.0
torch-scatter==2.1.0
torch-sparse==0.6.15
torch-spline-conv==1.2.1
torch-geometric==2.3.1
```

## 2. Usage

For example, if you want to run the Cora dataset (5 clinets), please run the following command to train our model:

```bash
python train_FedRGL_Cora.py --num_clients 5 --noisy_type uniform --noisy_rate 0.3
```

### 3. Filter Frequency

```bash
python train_FedRGL_Cora_Ten.py --num_clients 5 --noisy_type uniform --noisy_rate 0.3
```
