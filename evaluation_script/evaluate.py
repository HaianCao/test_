import numpy as np
import pandas as pd

# 6 attributes to predict
TARGET_COLS = [f"attr_{i}" for i in range(1, 7)]

def evaluate(
    Y_true: np.ndarray,
    Y_pred: np.ndarray,
) -> float:
    """
    Tính Exact Match Accuracy (phải đúng cả 6 thuộc tính cho một bản ghi).
    """
    Y_true = Y_true.astype(np.int64)
    Y_pred = np.round(Y_pred).astype(np.int64)

    # So sánh từng dòng (mỗi bản ghi) có khớp hoàn toàn cả 6 thuộc tính không
    exact_matches = np.all(Y_true == Y_pred, axis=1)
    
    # Tính accuracy (%)
    accuracy = np.mean(exact_matches) * 100.0
    return accuracy

def score(gold_path: str, pred_path: str) -> float:
    """
    Hàm tính điểm chính cho hệ thống.
    Trả về Exact Match %.
    """
    df_gold = pd.read_csv(gold_path)
    df_pred = pd.read_csv(pred_path)
    
    id_col = None
    if "id" in df_pred.columns:
        id_col = "id"
    
    if id_col is None:
        raise ValueError(f"Không tìm thấy cột ID trong file dự đoán. Các cột hiện có: {list(df_pred.columns)}")
        
    if id_col in df_gold.columns and id_col in df_pred.columns:
        df_gold = df_gold.sort_values(by=id_col).reset_index(drop=True)
        df_pred = df_pred.sort_values(by=id_col).reset_index(drop=True)
    
    if len(df_gold) != len(df_pred):
        raise ValueError(f"Số lượng dự đoán ({len(df_pred)}) KHÔNG khớp nhãn ({len(df_gold)})")
        
    Y_true = df_gold[TARGET_COLS].values
    Y_pred = df_pred[TARGET_COLS].values
        
    exact_match_acc = evaluate(Y_true, Y_pred)
        
    return exact_match_acc

if __name__ == "__main__":
    import sys
    if len(sys.argv) == 3:
        gold = sys.argv[1]
        pred = sys.argv[2]
        score_val = score(gold, pred)
        print(f"Final Score (Exact Match): {score_val:.2f}")
    else:
        print("Usage: python evaluate.py <gold_path> <pred_path>")
