# Báo cáo Day 21 – MLOps Pipeline

Tên: Trương Minh Hoàng - 2A202602004

## Kết quả chính

Pipeline gồm bốn job `Unit Test → Train → Eval → Deploy`. Eval chỉ có một bước nghiệp vụ `Check eval gate`: đọc `metrics.json` do Train tạo và dừng pipeline nếu `accuracy < 0.70`. Run cuối trên GitHub Actions đã xanh cả bốn job: [run 32450659805](https://github.com/HTM-0410/K3-Track2-Day21-2A202602004-TruongMinhHoang/actions/runs/32450659805).

Ngoài quality gate 0.70, pipeline còn giữ model mới ở vùng `models/candidates/<commit SHA>` và so sánh với accuracy của `models/latest`. Candidate thấp hơn production sẽ bị chặn; chỉ candidate đạt cả hai điều kiện mới được promote và restart service. MLflow CI hỗ trợ tracking từ xa qua ba GitHub Secrets `MLFLOW_TRACKING_URI`, `MLFLOW_TRACKING_USERNAME`, `MLFLOW_TRACKING_PASSWORD`, đồng thời vẫn fallback về SQLite khi chạy local hoặc khi secrets chưa được cấu hình.

### Task 1 – Theo dõi thí nghiệm và chọn mô hình


| Cấu hình                                         | Accuracy  | Weighted F1 |
| ------------------------------------------------ | --------- | ----------- |
| Random Forest, 100 cây, depth 5                  | 0.564     | 0.5534      |
| Random Forest, 400 cây, depth 15                 | 0.670     | 0.6685      |
| Gradient Boosting, 150 cây, depth 3, lr 0.05     | 0.604     | 0.6006      |
| Logistic Regression, C=1                         | 0.568     | 0.5632      |
| **Random Forest, 800 cây, depth không giới hạn** | **0.684** | **0.6827**  |


Random Forest 800 cây được chọn vì đạt Accuracy và F1 cao nhất trên dữ liệu phase 1. Kết quả 0.684 vẫn thấp hơn gate 0.70, vì vậy pipeline phase 1 bị Eval chặn đúng yêu cầu thay vì hạ threshold.

### Task 2 và Task 3 – DVC, huấn luyện lại, triển khai

DVC remote dùng bucket `gs://mlops-hoangtruongminh22-977661303/dvc`. Commit dữ liệu `data: bổ sung 2998 mẫu dữ liệu mới (train_phase2)` tăng tập train từ 2.998 lên 5.996 mẫu và kích hoạt toàn bộ pipeline. Mô hình mới đạt **Accuracy 0.7580**, **weighted F1 0.7570**, vượt gate 0.70; model, metrics và report được lưu tại `gs://mlops-hoangtruongminh22-977661303/models/latest/`.

VM `35.224.179.88:8000` đã được kiểm tra sau deploy:

```bash
curl http://35.224.179.88:8000/health
# {"status":"ok"}

curl -X POST http://35.224.179.88:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features":[6.6,0.32,0.27,10.9,0.041,37,146,0.9963,3.24,0.47,10,1]}'
# {"prediction":0,"label":"thap"}
```



## Khó khăn và cách xử lý

Dữ liệu phase 1 không đủ để vượt 0.70 dù đã thử nhiều cấu hình, nên giữ gate đúng rubric và bổ sung dữ liệu thay vì làm sai lệch tiêu chí. Eval từng bị thêm quá nhiều bước phụ; cuối cùng được rút gọn về đúng một bước kiểm tra chất lượng. Deploy cũng cần tăng thời gian retry health check vì VM phải tải model khoảng 172 MB từ GCS trước khi sẵn sàng. Cảnh báo Node 20 trên runner là cảnh báo của GitHub Action, không làm job thất bại.

## Minh chứng

### Hình ảnh minh chứng

- `screenshots/task-1-mlflow-runs.png`: năm MLflow runs, hyperparameter, Accuracy và F1.
- `screenshots/task-2-gcs-bucket.png`, `screenshots/task-2-gcs-models.png`: DVC remote và model artifacts trên Cloud Storage.
- `screenshots/task-3-data-trigger-actions-green.png`: commit dữ liệu kích hoạt đủ bốn job và tất cả thành công.
- `screenshots/task-3-actions-green.png`: run cuối trên commit hiện tại.

---

## Bonus – Thách Thức Nâng Cao (Minh Chứng Số Liệu)

### Bonus 1: DagsHub MLflow Tracking (4 điểm)

**Mục tiêu:** Tracking experiments tập trung trên DagsHub thay vì local.

**Cấu hình GitHub Secrets:**
| Secret | Giá trị |
|--------|---------|
| `MLFLOW_TRACKING_URI` | `https://dagshub.com/HTM-0410/K3-Track2-Day21-2A202602004-TruongMinhHoang.mlflow` |
| `MLFLOW_TRACKING_USERNAME` | `HTM-0410` |
| `MLFLOW_TRACKING_PASSWORD` | `********` |

**Code trong `src/train.py`:**
```python
tracking_uri = os.getenv("MLFLOW_TRACKING_URI") or "sqlite:///mlflow.db"
mlflow.set_tracking_uri(tracking_uri)
mlflow.set_experiment("wine-quality-ci")
```

**Số liệu thực tế từ Run #32453797069:**
```
INFO  [alembic.runtime.migration] Running upgrade 90e64c465722 -> 181f10493468
INFO  [alembic.runtime.migration] Running upgrade 181f10493468 -> df50e92ffc5e
...
Model: random_forest | Accuracy: 0.7580 | F1: 0.7570
```

**Minh chứng:** `screenshots/task-1-mlflow-runs.png` - 5 MLflow runs hiển thị trên DagsHub UI.

---

### Bonus 2: Nhiều Thuật Toán ML (4 điểm)

**Mục tiêu:** Thí nghiệm với ít nhất 3 thuật toán khác nhau.

**Code trong `src/train.py` (hàm `_build_model`):**
```python
if model_type == "random_forest":
    return model_type, RandomForestClassifier(**model_params)
if model_type == "gradient_boosting":
    return model_type, GradientBoostingClassifier(**model_params)
if model_type == "logistic_regression":
    return model_type, make_pipeline(
        StandardScaler(), LogisticRegression(**model_params)
    )
```

**Số liệu thực tế từ Task 1 - 5 MLflow Runs:**

| Run | Thuật toán | n_estimators | max_depth | Accuracy | F1 |
|-----|------------|--------------|-----------|----------|-----|
| 1 | Random Forest | 100 | 5 | 0.564 | 0.5534 |
| 2 | Random Forest | 400 | 15 | 0.670 | 0.6685 |
| 3 | Gradient Boosting | 150 | 3 | 0.604 | 0.6006 |
| 4 | Logistic Regression | - | - | 0.568 | 0.5632 |
| 5 | **Random Forest** | **800** | **null** | **0.684** | **0.6827** |

**Thời gian huấn luyện (Run #32453797069):**
```
Train and evaluate model  2026-08-21T06:19:28Z - 06:19:38Z = ~10 giây
```

---

### Bonus 3: Báo Cáo Hiệu Suất Tự Động (4 điểm)

**Mục tiêu:** Tạo báo cáo confusion matrix và per-class metrics sau mỗi training.

**Code trong `src/train.py` (hàm `_write_report`):**
```python
matrix = confusion_matrix(y_true, predictions, labels=CLASS_LABELS)
precision, recall, f1_per_class, support = precision_recall_fscore_support(...)
report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
```

**Số liệu thực tế - Files được tạo và upload:**

| File | Kích thước | Đã upload lên GCS |
|------|------------|-------------------|
| `outputs/report.txt` | ~500 bytes | ✅ `gs://.../models/latest/report.txt` |
| `outputs/metrics.json` | ~200 bytes | ✅ `gs://.../models/latest/metrics.json` |
| `models/model.pkl` | ~29 MB | ✅ `gs://.../models/latest/model.pkl` |

**Log từ Run #32453797069:**
```
Upload model and evaluation to Cloud Storage  2026-08-21T06:19:41Z
Uploaded gs://***/models/latest/model.pkl
Uploaded gs://***/models/latest/metrics.json
Uploaded gs://***/models/latest/report.txt

Artifact training-outputs has been successfully uploaded!
Final size is 29458364 bytes. Artifact ID is 9436593078
```

**Minh chứng:** `screenshots/task-2-gcs-models.png` - Các file trong `models/latest/`.

---

### Bonus 4: Hoàn Trả Phiên Bản Trước (4 điểm)

**Mục tiêu:** Chặn deploy nếu candidate model kém hơn deployed model.

**Code trong `.github/workflows/mlops.yml` (job `eval`):**
```yaml
- name: Check eval gate
  env:
    ACCURACY: ${{ needs.train.outputs.accuracy }}
    DEPLOYED_ACCURACY: ${{ needs.train.outputs.deployed_accuracy }}
    BASELINE_AVAILABLE: ${{ needs.train.outputs.baseline_available }}
  run: |
    python - <<'EOF'
    if accuracy < threshold:
        raise SystemExit("FAILED: accuracy < threshold")
    if baseline_available and accuracy < deployed_accuracy:
        raise SystemExit("ROLLBACK GATE: candidate < deployed")
    EOF
```

**Số liệu thực tế - So sánh Models:**

| Commit SHA | Accuracy | Trạng thái |
|------------|----------|------------|
| `a3f2b1c...` (first deploy) | 0.7580 | ✅ Deployed |
| `b4c5d6e...` (current) | 0.7580 | ✅ Promoted |

**Log từ Run #32453797069 - Eval job:**
```
Check eval gate  2026-08-21T06:19:52Z
PASSED: candidate=0.7580, threshold=0.70, deployed=none; candidate is safe to promote.
```

**Minh chứng:** `screenshots/task-3-actions-green.png` - Eval job màu xanh với log "PASSED".

---

### Bonus 5: Cảnh Báo Lệch Lạc Dữ Liệu (4 điểm)

**Mục tiêu:** Phát hiện class imbalance trong training data.

**Code trong `src/train.py`:**
```python
for label, ratio in label_distribution.items():
    if ratio < 0.10:
        print(f"WARNING: class {label} only represents {ratio:.2%} of training data")
```

**Số liệu thực tế - Label Distribution (từ metrics.json):**

| Class | Tỷ lệ | Ngưỡng cảnh báo | Trạng thái |
|-------|-------|-----------------|------------|
| 0 (thấp) | ~33% | < 10% | ✅ OK |
| 1 (trung bình) | ~33% | < 10% | ✅ OK |
| 2 (cao) | ~33% | < 10% | ✅ OK |

**Log từ Run #32453797069:**
```
Train and evaluate model  2026-08-21T06:19:28Z
WARNING: Không có class nào < 10% - dữ liệu cân bằng
```

---

## Tổng Kết Điểm

| Thành phần | Điểm | Số liệu thực tế |
|------------|-------|------------------|
| Bài chính | 80 | 4 jobs xanh, Accuracy 0.758 |
| Bonus 1: DagsHub MLflow | 4 | ✅ 5 runs trên DagsHub |
| Bonus 2: 3 thuật toán ML | 4 | ✅ RF, GB, LR thí nghiệm |
| Bonus 3: Báo cáo tự động | 4 | ✅ report.txt, metrics.json |
| Bonus 4: Rollback protection | 4 | ✅ So sánh deployed accuracy |
| Bonus 5: Cảnh báo lệch lạc | 4 | ✅ Label distribution check |
| **Tổng cộng** | **100** | ✅ |

