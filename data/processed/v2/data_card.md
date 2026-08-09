# VietNews v2 — data card
- nguồn: nam194/vietnews (HuggingFace), 3 split theo file gốc; replay v1 (143.811)
- tổng dòng: 143811
- original_split: {"train": 99130, "test": 22497, "validation": 22184}
- split_v2 (65/15/20, seed 42): {"train": 93471, "test": 28762, "validation": 21578}
- cột: id, original_split, split_v2, guid, title, title_norm, abstract, abstract_norm, article, article_norm, n_tokens_article, n_tokens_abstract
- seed: 42
- ngưỡng đánh dấu: {"flag_empty": true, "min_abstract_chars": 20, "max_abstract_to_article": 0.75, "near_dup_jaccard": 0.9, "max_article_chars": 20000, "max_abstract_chars": 2000, "near_dup_cap_group": 50, "cut_512": 512, "cut_1024": 1024, "cut_cover_min": 0.9}
- review_approved: True (approval_by: Nguoi dung)
- PII đã mask 1124 ô (policy=pii->mask)
- review_queue: {"audit_flags.csv": 41, "fact_flags.csv": 76098, "label_flags.csv": 2, "near_duplicates.csv": 68, "near_duplicates_cross_split.csv": 108, "pii.csv": 1124}
- quy tắc: không cắt article lúc lưu; test v2 khóa, không dùng chọn tham số
- rủi ro: abstract do con người viết có thể chứa lỗi nhãn; model có thể sinh sai số liệu/tên;
  near-dup chỉ là sàng lọc (xem manifest.near_dup_*_scan).
