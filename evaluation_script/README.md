# Đánh giá (Evaluation)

## Hướng dẫn sử dụng (Usage)

Script đánh giá tính toán và trả về điểm **Exact Match Accuracy**.

Cài đặt các thư viện cần thiết:
```bash 
pip install -r requirements.txt
```

### Chạy từ Command Line

```bash
python evaluate.py sample_data/Y_gold.csv sample_data/Y_pred.csv
```

### Chạy trên Python / Jupyter Notebook

```python
from evaluate import score

# Đánh giá file trực tiếp và lấy điểm Exact Match Accuracy
score = score('sample_data/Y_gold.csv', 'sample_data/Y_pred.csv')
print("Kết quả Exact Match Accuracy:", score)
```
