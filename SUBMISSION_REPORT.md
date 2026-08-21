# Báo cáo Day 21 – MLOps Pipeline

## Kết quả chính

Pipeline gồm bốn job `Unit Test → Train → Eval → Deploy`. Eval chỉ có một bước nghiệp vụ `Check eval gate`: đọc `metrics.json` do Train tạo và dừng pipeline nếu `accuracy < 0.70`. Run cuối trên GitHub Actions đã xanh cả bốn job: [run 32450659805](https://github.com/HTM-0410/K3-Track2-Day21-2A202602004-TruongMinhHoang/actions/runs/32450659805).

### Task 1 – Theo dõi thí nghiệm và chọn mô hình

| Cấu hình | Accuracy | Weighted F1 |
|---|---:|---:|
| Random Forest, 100 cây, depth 5 | 0.564 | 0.5534 |
| Random Forest, 400 cây, depth 15 | 0.670 | 0.6685 |
| Gradient Boosting, 150 cây, depth 3, lr 0.05 | 0.604 | 0.6006 |
| Logistic Regression, C=1 | 0.568 | 0.5632 |
| **Random Forest, 800 cây, depth không giới hạn** | **0.684** | **0.6827** |

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

- `screensot/task-1-mlflow-runs.png`: năm MLflow runs, hyperparameter, Accuracy và F1.
- `screensot/task-2-gcs-bucket.png`, `screensot/task-2-gcs-models.png`: DVC remote và model artifacts trên Cloud Storage.
- `screensot/task-3-data-trigger-actions-green.png`: commit dữ liệu kích hoạt đủ bốn job và tất cả thành công.
- `screensot/task-3-actions-green.png`: run cuối trên commit hiện tại.
