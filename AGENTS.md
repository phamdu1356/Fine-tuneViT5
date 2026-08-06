# AGENTS.md

## Phạm vi và mục tiêu

- Các quy tắc này áp dụng cho toàn bộ repository.
- Mục tiêu của dự án là xây dựng quy trình fine-tune và đánh giá ViT5 có thể tái lập, dễ kiểm tra và an toàn với dữ liệu.
- Repository đang ở giai đoạn khởi tạo. Luôn đọc cấu trúc và các file cấu hình hiện có trước khi giả định framework, phiên bản Python hoặc câu lệnh chạy.

## Cách làm việc

- Trao đổi và viết tài liệu bằng tiếng Việt, trừ khi người dùng yêu cầu ngôn ngữ khác. Tên biến, hàm, lớp và module dùng tiếng Anh rõ nghĩa.
- Chỉ thay đổi phần liên quan trực tiếp đến yêu cầu; không tự ý refactor hoặc định dạng lại file ngoài phạm vi.
- Giữ nguyên thay đổi chưa commit của người dùng. Không commit, push, phát hành model hoặc tải dữ liệu lên dịch vụ bên ngoài nếu chưa được yêu cầu rõ ràng.
- Trước khi sửa, kiểm tra `README.md`, manifest dependency, cấu hình train/eval và test hiện có. Không bịa câu lệnh setup, train hoặc test.
- Khi yêu cầu còn thiếu chi tiết nhưng có thể dùng giả định an toàn, nêu rõ giả định và ưu tiên thay đổi nhỏ, có thể đảo ngược.

## Python và cấu trúc mã nguồn

- Dùng virtual environment cục bộ (ưu tiên `.venv`) và dependency được khai báo, khóa phiên bản phù hợp trong manifest của dự án. Không phụ thuộc package cài global.
- Ưu tiên `pathlib`, type hints, logging thay cho `print` trong mã chạy chính, và UTF-8 cho file văn bản.
- Không hard-code đường dẫn máy cá nhân, token, tên dataset hoặc tham số huấn luyện trong mã nguồn. Đưa chúng vào CLI hoặc file cấu hình có giá trị mặc định hợp lý.
- Tách logic có thể tái sử dụng khỏi notebook. Notebook chỉ dùng cho khám phá, minh họa hoặc phân tích kết quả.
- Khi cần mở rộng cấu trúc, ưu tiên `src/` cho mã nguồn, `configs/` cho cấu hình, `scripts/` cho entry point, `tests/` cho kiểm thử và `notebooks/` cho thử nghiệm tương tác.

## Dữ liệu và huấn luyện

- Xem dữ liệu đầu vào là bất biến. Không chỉnh sửa trực tiếp file dữ liệu gốc; tạo bước tiền xử lý có thể chạy lại.
- Không commit dataset riêng tư, dữ liệu lớn, checkpoint, tensorboard log, cache hoặc artifact sinh ra. Cập nhật `.gitignore` khi thêm loại output mới.
- Ngăn rò rỉ dữ liệu: tạo train/validation/test split độc lập; mọi phép fit vocabulary, thống kê hoặc biến đổi học từ dữ liệu chỉ dùng train split.
- Cố định và ghi lại seed cho Python, NumPy, PyTorch và framework liên quan khi có thể. Ghi lại phiên bản dataset, model/tokenizer gốc, dependency, hyperparameter và commit nguồn cho mỗi lần chạy quan trọng.
- Cấu hình rõ độ dài input/target, truncation, padding và batch size. Nếu dùng Hugging Face seq2seq, token padding trong labels phải được loại khỏi loss (thường bằng `-100`).
- Lưu tokenizer, cấu hình và training arguments cùng checkpoint. Luồng resume phải tiếp tục đúng optimizer, scheduler, global step và seed state nếu framework hỗ trợ.
- Không tự động chạy full training hoặc tải model/dataset lớn. Trước tiên chạy smoke test trên tập rất nhỏ với tối đa vài step, sau đó mới chạy dài khi người dùng yêu cầu.
- Đánh giá trên validation/test tách biệt và báo cả metric chất lượng lẫn thông tin thực thi cần thiết như thời gian, thiết bị, batch size và mức dùng bộ nhớ khi có liên quan.

## Kiểm tra thay đổi

- Dùng đúng formatter, linter và test runner đã được khai báo trong repository. Nếu chưa có, không tự áp đặt một toolchain lớn chỉ để sửa nhỏ.
- Với thay đổi Python, ít nhất kiểm tra import/cú pháp cho phần bị ảnh hưởng. Với logic xử lý dữ liệu, thêm test nhỏ cho dữ liệu rỗng, Unicode tiếng Việt, chuỗi dài và giá trị thiếu khi phù hợp.
- Với thay đổi train/eval, chạy smoke test end-to-end trên dữ liệu mẫu nhỏ và xác nhận có thể tokenize, forward, tính loss, evaluate và lưu/load checkpoint theo phạm vi thay đổi.
- Không coi kết quả notebook cũ hoặc checkpoint có sẵn là bằng chứng rằng mã hiện tại hoạt động.
- Khi hoàn tất, báo ngắn gọn file đã đổi, lệnh đã chạy, kết quả thực tế và phần nào chưa thể xác minh.

## Tài liệu và Git

- Khi thêm hoặc đổi workflow, cập nhật `README.md` bằng câu lệnh có thể sao chép để setup, train, evaluate và inference; mô tả input/output và vị trí artifact.
- Không đưa secret vào Git. Dùng biến môi trường và cung cấp file ví dụ như `.env.example` chỉ chứa placeholder.
- Tránh commit file nhị phân lớn. Chỉ dùng Git LFS khi repository đã chọn LFS hoặc người dùng yêu cầu.
- Không viết lại lịch sử Git, xóa branch/tag, hoặc thực hiện thao tác phá hủy nếu chưa có yêu cầu rõ ràng.

## Quy tắc review

- Ưu tiên phát hiện lỗi gây rò rỉ train/test, mapping sai cặp input-target, padding labels sai, metric tính trên dữ liệu train, model không chuyển đúng `train()`/`eval()`, sai device/dtype và checkpoint không thể resume.
- Kiểm tra thay đổi có làm mất khả năng tái lập, ghi đè dữ liệu/checkpoint, lộ secret hoặc âm thầm tải/chạy tài nguyên lớn hay không.
- Mọi nhận định về cải thiện chất lượng hoặc tốc độ phải dựa trên phép đo có cùng dữ liệu, split, seed, phần cứng và cấu hình so sánh.
