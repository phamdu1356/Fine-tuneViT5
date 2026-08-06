# Phân công công việc nhóm Fine-tune ViT5

## 1. Mục tiêu chung

Nhóm xây dựng một quy trình fine-tune `VietAI/vit5-base` cho bài toán tóm tắt văn bản tiếng Việt, từ dữ liệu thô đến đánh giá cuối cùng. Kết quả phải có thể chạy lại, kiểm tra được nguồn gốc dữ liệu và so sánh công bằng giữa model trước và sau fine-tune.

### Điều kiện hoàn thành chung

- Dataset được làm sạch, chia tập và đóng băng thành phiên bản `v1`.
- Model gốc được đánh giá trên test set trước khi fine-tune.
- Quá trình fine-tune có cấu hình, seed, log và checkpoint có thể tải lại.
- Model sau fine-tune được đánh giá bằng đúng test set và quy trình của baseline.
- Có báo cáo cuối cùng chỉ rõ mức cải thiện, hạn chế và hướng thử nghiệm tiếp theo.

## 2. Thành viên và trách nhiệm chính

| Thành viên | Trách nhiệm chính | Bàn giao cho |
| --- | --- | --- |
| Duy Anh | Kiểm tra, làm sạch và chuẩn hóa dữ liệu thô ban đầu | Minh Anh |
| Minh Anh | Hoàn thiện, chia tập và đóng băng dataset `v1` | Hải Anh, Giang và Khải |
| Hải Anh | Đánh giá `VietAI/vit5-base` trước khi fine-tune | Giang và Khải |
| Giang | Chuẩn bị môi trường, fine-tune và đóng gói model | Khải |
| Khải | Đánh giá model sau fine-tune và lập báo cáo so sánh | Cả nhóm |

## 3. Quy ước dữ liệu và artifact

### Schema dữ liệu thống nhất

Mỗi mẫu dữ liệu phải có tối thiểu các trường sau:

| Trường | Kiểu | Yêu cầu |
| --- | --- | --- |
| `id` | string | Duy nhất, ổn định giữa các lần xử lý |
| `document` | string | Văn bản nguồn cần tóm tắt, không được rỗng |
| `summary` | string | Tóm tắt tham chiếu, không được rỗng |
| `source` | string | Nguồn hoặc bộ dữ liệu gốc nếu xác định được |
| `group_id` | string | Gom các bài trùng/gần trùng để chống rò rỉ khi chia tập |
| `metadata` | object | Thông tin bổ sung như ngày, chủ đề hoặc giấy phép nếu có |

### Vị trí artifact dự kiến

| Loại | Vị trí |
| --- | --- |
| Cấu hình dữ liệu, train và evaluation | `configs/` |
| Báo cáo kiểm tra và kết quả | `reports/` |
| Prediction, log, checkpoint và dữ liệu sinh ra | `outputs/` |

`outputs/`, dataset và checkpoint lớn không được commit trực tiếp vào Git. Trước khi sinh artifact, người phụ trách phải kiểm tra `.gitignore`. Secret, access token và dữ liệu nhạy cảm không được đưa vào repository.

## 4. Công việc chi tiết

### 4.1. Duy Anh — kiểm tra và làm sạch dữ liệu thô

**Đầu vào:** Toàn bộ dữ liệu thô mà nhóm đang có.

#### Checklist công việc

- [ ] Lập danh sách tất cả file dữ liệu, nguồn tải, định dạng, dung lượng và số lượng mẫu.
- [ ] Kiểm tra giấy phép sử dụng, điều kiện chia sẻ và ghi rõ dữ liệu nào không được đưa lên Git.
- [ ] Kiểm tra dữ liệu có chứa email, số điện thoại, địa chỉ, mã định danh hoặc thông tin nhạy cảm hay không.
- [ ] Chuyển dữ liệu về schema chung gồm `id`, `document`, `summary`, `source`, `group_id` và `metadata`.
- [ ] Kiểm tra `id` bị thiếu hoặc trùng; tạo `id` ổn định nếu dữ liệu gốc chưa có.
- [ ] Thống kê số mẫu thiếu `document`, thiếu `summary`, sai kiểu dữ liệu hoặc có nội dung rỗng sau khi trim.
- [ ] Phát hiện văn bản không phải tiếng Việt hoặc chứa quá nhiều ký tự lỗi.
- [ ] Phát hiện mẫu trùng chính xác và gần trùng ở cả `document` lẫn `summary`.
- [ ] Kiểm tra các cặp document-summary bất thường: summary dài hơn document, không liên quan hoặc có dấu hiệu bị đảo cột.
- [ ] Xây dựng quy tắc làm sạch có thể chạy lại; không sửa trực tiếp dữ liệu gốc.
- [ ] Lưu thống kê số mẫu trước làm sạch, số bị loại theo từng lý do và số mẫu còn lại.
- [ ] Chọn một tập mẫu ngẫu nhiên để Minh Anh kiểm tra chéo chất lượng làm sạch.

#### Đầu ra bàn giao

- `reports/data_audit.md`: nguồn, giấy phép, thống kê lỗi và quyết định xử lý.
- `configs/data_cleaning.yaml`: toàn bộ quy tắc làm sạch đã sử dụng.
- Dữ liệu sạch sơ bộ kèm danh sách mẫu bị loại và lý do loại.
- Hướng dẫn chạy lại quá trình làm sạch từ dữ liệu thô.

#### Tiêu chí hoàn thành

- Dữ liệu gốc không bị ghi đè.
- Mỗi mẫu đầu ra đúng schema và có `id` duy nhất.
- Mọi quy tắc loại hoặc sửa mẫu đều được ghi lại.
- Minh Anh có thể chạy lại quy trình và nhận được cùng số mẫu, cùng checksum đầu ra.

---

### 4.2. Minh Anh — xử lý và đóng băng dataset

**Đầu vào:** Dữ liệu sạch sơ bộ và báo cáo audit từ Duy Anh.

#### Checklist công việc

- [ ] Review một tập mẫu ngẫu nhiên và phản hồi các lỗi làm sạch còn sót cho Duy Anh.
- [ ] Chuẩn hóa Unicode, dấu tiếng Việt, khoảng trắng, xuống dòng và ký tự điều khiển.
- [ ] Xác nhận `document` và `summary` vẫn ghép đúng cặp sau toàn bộ bước xử lý.
- [ ] Dùng tokenizer đúng revision của `VietAI/vit5-base` để thống kê độ dài input và target.
- [ ] Báo cáo tỷ lệ mẫu bị cắt ứng với các giá trị `max_source_length` và `max_target_length` dự kiến.
- [ ] Nhóm các mẫu trùng/gần trùng bằng `group_id` trước khi chia tập.
- [ ] Chia train/validation/test theo tỷ lệ mặc định 80/10/10 với seed `42`.
- [ ] Giữ phân bố nguồn, chủ đề và độ dài gần tương đương giữa ba tập khi dữ liệu cho phép.
- [ ] Xác nhận một `group_id` không xuất hiện ở nhiều split và không có nội dung trùng chéo.
- [ ] Tạo manifest chứa số mẫu, checksum file, seed, tỷ lệ split và phiên bản script/config.
- [ ] Viết data card mô tả nguồn, giấy phép, preprocessing, thống kê và hạn chế của dataset.
- [ ] Đóng băng dataset thành phiên bản `v1`; mọi thay đổi tiếp theo phải tạo phiên bản mới.
- [ ] Cung cấp danh sách test IDs cho Hải Anh và Khải; không công khai nhãn test cho luồng train.

#### Đầu ra bàn giao

- Dataset `v1` gồm train, validation và test.
- `configs/data_v1.yaml`: schema, seed, tỷ lệ chia và tham số xử lý.
- `reports/data_card_v1.md`: mô tả đầy đủ dataset.
- Manifest và checksum để mọi thành viên xác nhận đang dùng cùng phiên bản.

#### Tiêu chí hoàn thành

- Tỷ lệ chia là 80/10/10 hoặc có giải thích rõ nếu phải thay đổi.
- Không có `id`, `group_id` hoặc mẫu gần trùng bị rò rỉ giữa các split.
- Các checksum được ghi lại và kiểm tra thành công.
- Duy Anh xác nhận quy trình dữ liệu; Hải Anh và Giang xác nhận đọc được dataset `v1`.

---

### 4.3. Hải Anh — đánh giá model trước khi train

**Đầu vào:** Dataset `v1` từ Minh Anh và checkpoint gốc `VietAI/vit5-base`.

#### Checklist công việc

- [ ] Ghi lại model ID, revision, tokenizer revision và phiên bản thư viện sử dụng.
- [ ] Xây dựng script inference/evaluation có thể chạy lại từ command line.
- [ ] Thử prompt, preprocessing và decoding trên validation set; tuyệt đối không tối ưu bằng test set.
- [ ] Khóa cấu hình evaluation gồm prompt, normalization, max length, beam size và batch size.
- [ ] Chạy model gốc trên toàn bộ test set một lần bằng cấu hình đã khóa.
- [ ] Tính ROUGE-1, ROUGE-2 và ROUGE-L; ghi rõ thư viện và phiên bản metric.
- [ ] Lưu prediction theo từng `id`, kèm summary tham chiếu để Khải có thể đối chiếu.
- [ ] Ghi lại phần cứng, thời gian chạy, batch size, precision, throughput và bộ nhớ sử dụng nếu đo được.
- [ ] Chọn cố định tối thiểu 30 mẫu đại diện cho đánh giá định tính trước/sau train.
- [ ] Chấm các mẫu theo độ bao phủ, mạch lạc, lặp ý, sai sự thật và hallucination.
- [ ] Tổng hợp lỗi phổ biến của model gốc và đề xuất điểm Giang cần theo dõi khi train.
- [ ] Bàn giao toàn bộ cấu hình evaluation cho Khải; sau khi bàn giao không tự ý thay đổi cấu hình.

#### Đầu ra bàn giao

- `configs/eval_baseline.yaml`: cấu hình evaluation đã khóa.
- `outputs/baseline/predictions.jsonl`: prediction của model gốc theo từng `id`.
- `reports/baseline_evaluation.md`: metric, tốc độ và phân tích định tính.
- Danh sách 30 test IDs cố định dùng cho phân tích trước/sau train.

#### Tiêu chí hoàn thành

- Baseline chạy trên đúng checksum của test set `v1`.
- Một thành viên khác có thể chạy lại evaluation từ cấu hình đã bàn giao.
- Metric, prediction và thông tin môi trường đầy đủ.
- Giang và Khải xác nhận đã nhận cùng một cấu hình evaluation.

---

### 4.4. Giang — fine-tune và xử lý model

**Đầu vào:** Train/validation của dataset `v1` và báo cáo baseline từ Hải Anh.

#### Checklist công việc

- [ ] Kiểm kê phần cứng thực tế: GPU, VRAM, RAM, CPU, dung lượng đĩa, CUDA và driver.
- [ ] Ghi lại phiên bản Python, PyTorch, Transformers, Datasets, tokenizer và các dependency chính.
- [ ] Chọn `fp32`, `fp16` hoặc `bf16` dựa trên khả năng phần cứng; không giả định khi chưa kiểm tra.
- [ ] Xác định per-device batch size, gradient accumulation và effective batch size.
- [ ] Chuẩn bị cấu hình train có model revision, seed, learning rate, epoch/max steps, warmup và checkpoint policy.
- [ ] Kiểm tra labels padding được loại khỏi loss, thường bằng giá trị `-100` khi dùng Hugging Face seq2seq.
- [ ] Chạy smoke test trên 16–32 mẫu trong tối đa vài step.
- [ ] Xác nhận smoke test có forward, backward, validation, save checkpoint, load checkpoint và inference thành công.
- [ ] Kiểm tra loss hữu hạn; dừng nếu xuất hiện `NaN`, hết bộ nhớ hoặc mapping input-target sai.
- [ ] Fine-tune chỉ bằng train set và chọn hyperparameter/checkpoint bằng validation set.
- [ ] Không đọc metric test để quyết định learning rate, epoch, prompt hoặc checkpoint.
- [ ] Theo dõi train/validation loss, ROUGE validation, learning rate, thời gian và tài nguyên.
- [ ] Lưu đủ checkpoint để resume sau khi phiên chạy bị gián đoạn.
- [ ] Chọn checkpoint tốt nhất theo validation ROUGE-L và ghi rõ quy tắc xử lý khi bằng điểm.
- [ ] Tải lại checkpoint tốt nhất trong một process mới và chạy inference kiểm tra.
- [ ] Đóng gói model, tokenizer, generation config, training config và hướng dẫn sử dụng.

#### Đầu ra bàn giao

- `reports/hardware_and_training.md`: phần cứng, môi trường, thí nghiệm và kết quả validation.
- `configs/train_v1.yaml`: cấu hình đầy đủ của lần train được chọn.
- `outputs/checkpoints/best/`: model, tokenizer và generation config tốt nhất.
- Training log, seed, commit nguồn và lệnh dùng để train/resume/inference.

#### Tiêu chí hoàn thành

- Smoke test hoàn thành trước khi chạy train dài.
- Lần train có thể tái tạo từ dataset manifest và config đã ghi.
- Checkpoint tốt nhất tải lại và inference thành công độc lập.
- Khải nhận đủ model, tokenizer, config và hướng dẫn mà không cần hỏi thêm tham số ẩn.

---

### 4.5. Khải — đánh giá và so sánh sau train

**Đầu vào:** Checkpoint tốt nhất từ Giang, test set `v1` và cấu hình baseline từ Hải Anh.

#### Checklist công việc

- [ ] Xác nhận checksum test set khớp với lần đánh giá baseline.
- [ ] Tải checkpoint và tokenizer trong một môi trường sạch; kiểm tra inference không phụ thuộc file cục bộ bị thiếu.
- [ ] Dùng đúng preprocessing, prompt, decoding và metric config đã khóa bởi Hải Anh.
- [ ] Chạy model fine-tuned trên cùng toàn bộ test IDs của baseline.
- [ ] Tính ROUGE-1, ROUGE-2 và ROUGE-L bằng cùng thư viện, phiên bản và normalization.
- [ ] Lập bảng điểm baseline, fine-tuned, chênh lệch tuyệt đối và phần trăm thay đổi.
- [ ] So sánh thời gian suy luận, throughput và tài nguyên trên cùng phần cứng nếu có thể.
- [ ] Đối chiếu 30 mẫu định tính cố định trước và sau train.
- [ ] Phân loại trường hợp cải thiện, suy giảm, thiếu ý, dư ý, lặp nội dung và hallucination.
- [ ] Kiểm tra các mẫu có ROUGE cao nhưng chất lượng thực tế thấp hoặc ngược lại.
- [ ] Nêu rõ giới hạn của dataset, metric và phần cứng khi diễn giải kết quả.
- [ ] Kết luận model có cải thiện đủ rõ để sử dụng hay cần thêm vòng thử nghiệm.
- [ ] Đề xuất tối đa ba thử nghiệm tiếp theo, mỗi thử nghiệm kèm giả thuyết và metric theo dõi.

#### Đầu ra bàn giao

- `outputs/final_eval/predictions.jsonl`: prediction của model fine-tuned theo từng `id`.
- `reports/final_comparison.md`: bảng metric, phân tích định tính và kết luận.
- Bảng đối chiếu baseline/fine-tuned có đủ cấu hình và thông tin môi trường.
- Danh sách lỗi nổi bật và đề xuất vòng thử nghiệm tiếp theo.

#### Tiêu chí hoàn thành

- Baseline và model fine-tuned chỉ khác checkpoint; test set và quy trình evaluation phải giống nhau.
- Báo cáo phân biệt rõ kết quả đo được, nhận xét định tính và giả thuyết.
- Mọi con số trong bảng có thể truy ngược tới config, prediction và dataset manifest.
- Cả nhóm review và thống nhất kết luận cuối cùng.

## 5. Năm cổng nghiệm thu và bàn giao

| Cổng | Chủ trì | Người kiểm tra/nhận | Điều kiện được thông qua |
| --- | --- | --- | --- |
| G1 — Data audit | Duy Anh | Minh Anh | Có báo cáo nguồn, lỗi, quy tắc làm sạch và dữ liệu sạch sơ bộ có thể tái tạo |
| G2 — Dataset freeze | Minh Anh | Duy Anh, Hải Anh, Giang | Dataset `v1` có manifest/checksum, không rò rỉ split và được đọc thành công |
| G3 — Baseline freeze | Hải Anh | Giang, Khải | Evaluation config được khóa; baseline metric và prediction đầy đủ |
| G4 — Training complete | Giang | Khải | Checkpoint tốt nhất tải lại được; config, log và hướng dẫn đã bàn giao |
| G5 — Final evaluation | Khải | Cả nhóm | So sánh công bằng, báo cáo đầy đủ và kết luận được nhóm review |

Không được bỏ qua cổng trước để công bố kết quả của cổng sau. Giang có thể chuẩn bị môi trường và Hải Anh có thể xây dựng evaluation script trong lúc dataset đang được xử lý, nhưng baseline chính thức và full training chỉ bắt đầu sau G2.

## 6. Quy tắc phối hợp chung

- Mỗi thay đổi dataset, config hoặc checkpoint phải có version và người tạo.
- Mỗi lần chạy quan trọng phải ghi seed, commit nguồn, dataset checksum, model revision và câu lệnh thực thi.
- Không tự ý thay đổi test set hoặc evaluation config sau G3. Nếu buộc phải thay đổi, baseline và model fine-tuned phải được chạy lại cùng cấu hình mới.
- Không dùng test set để chọn prompt, hyperparameter, epoch hoặc checkpoint.
- Không ghi đè artifact của người khác; tạo version hoặc thư mục run mới.
- Mọi lỗi chặn bàn giao phải được ghi rõ: lỗi gì, cách tái hiện, người đang xử lý và artifact bị ảnh hưởng.
- Chỉ công bố kết quả khi có đường dẫn tới báo cáo, config và prediction tương ứng.

## 7. Mẫu cập nhật tiến độ

Mỗi thành viên dùng mẫu sau khi cập nhật công việc:

```text
Người phụ trách:
Công việc đã hoàn thành:
Artifact/đường dẫn:
Kiểm tra đã chạy:
Kết quả:
Vấn đề đang chặn:
Người cần nhận bàn giao:
Việc tiếp theo:
```

Tài liệu hiện không gắn ngày hoàn thành vì nhóm chưa chốt lịch dự án. Khi có deadline, bổ sung mốc thời gian cho từng cổng G1–G5 thay vì đặt deadline rời rạc cho từng checklist.
