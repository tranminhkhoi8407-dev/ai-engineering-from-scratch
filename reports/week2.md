# Phase 05 — Báo cáo học tập 4 bài đầu

**Phạm vi:** Bài 01–04 · Text Processing, Bag of Words/TF-IDF, Word2Vec, GloVe/FastText/Subword  
**Môi trường chạy:** Python 3.12.10 trong `.venv` của repo  
**Cách làm:** chạy code mẫu của từng bài trong notebook tại chính thư mục `notebook/`, sau đó chạy phần thực hành tự viết. Output được lưu trong các notebook để người học tự chụp ảnh khi cần.

> Phase 05 có 29 bài trong repo. Báo cáo này chỉ bao gồm bốn bài đầu theo yêu cầu hiện tại.

## Tóm tắt theo bài

### 1. Text Processing — Tokenization, Stemming, Lemmatization

[Notebook Bài 01](../phases/05-nlp-foundations-to-advanced/01-text-processing/notebook/01-text-processing.ipynb) · [Tài liệu](../phases/05-nlp-foundations-to-advanced/01-text-processing/docs/en.md)

Tiền xử lý đưa chuỗi văn bản về những đơn vị mà mô hình có thể xử lý. Tokenizer xác định ranh giới token; stemmer cắt hậu tố bằng luật; lemmatizer dùng thêm thông tin từ loại và hình thái để đưa từ về dạng từ điển. Ba bước này không thể thay thế cho nhau: stemmer có thể tạo một chuỗi không phải từ, trong khi lemmatizer có thể giữ nghĩa tốt hơn nếu có bảng từ và POS phù hợp.

Code mẫu đã chạy câu `The cats were running at 3pm.` và in ra `running → run` ở phần lemma, `were → be`, còn số `3pm` tách thành `3` và `pm`. Notebook thực hành mở rộng lên câu mới: stemmer cho `ponies → poni`, trong khi lookup lemma cho `ponies → pony`; từ `watched` không đổi vì bảng/fallback nhỏ chưa bao phủ hậu tố `-ed`.

Điểm cần nhớ là tokenizer trong code bài dùng chữ cái Latin `A–Z`, nên không phải tokenizer tiếng Việt hoàn chỉnh. Nếu huấn luyện với một pipeline rồi triển khai pipeline khác, mô hình sẽ nhận phân phối token khác và chất lượng có thể giảm.

### 2. Bag of Words, TF-IDF và Text Representation

[Notebook Bài 02](../phases/05-nlp-foundations-to-advanced/02-bag-of-words-tfidf/notebook/02-bag-of-words-tfidf.ipynb) · [Tài liệu](../phases/05-nlp-foundations-to-advanced/02-bag-of-words-tfidf/docs/en.md)

Bag of Words tạo vocabulary và đếm số lần mỗi từ xuất hiện trong tài liệu; thứ tự từ bị bỏ. TF-IDF thay trọng số đếm bằng tích giữa term frequency và inverse document frequency. Trong code bài, IDF có smoothing để tránh chia cho 0. Chuẩn hóa L2 đưa vector về cùng thang độ dài, nhờ vậy cosine similarity tập trung vào hướng của vector hơn là độ dài tài liệu.

Code mẫu tạo ma trận TF-IDF cho ba câu và ma trận cosine: hai câu có chung `cat`, `sat`, `on`, `mat` đạt độ tương tự `0.80`; hai câu ít từ chung đạt `0.46`. Thực hành riêng dùng corpus ticket hỗ trợ. Query `refund delayed bank transfer` xếp ticket về refund bị trễ lên đầu với cosine `0.738`.

BoW/TF-IDF dễ kiểm tra và thường là baseline nhanh cho phân loại hoặc tìm kiếm từ khóa. Chúng không nắm ý nghĩa đồng nghĩa và mất thứ tự. Notebook thêm bigram để thể hiện `not_good`; tuy vậy bigram cũng không giải quyết đầy đủ cấu trúc phủ định hay ngữ cảnh.

### 3. Word Embeddings — Word2Vec from Scratch

[Notebook Bài 03](../phases/05-nlp-foundations-to-advanced/03-word-embeddings-word2vec/notebook/03-word-embeddings-word2vec.ipynb) · [Tài liệu](../phases/05-nlp-foundations-to-advanced/03-word-embeddings-word2vec/docs/en.md)

Word2Vec dựa trên giả thuyết phân bố: từ xuất hiện trong ngữ cảnh tương tự có thể có biểu diễn gần nhau. Skip-gram tạo các cặp từ trung tâm/ngữ cảnh trong một cửa sổ. Negative sampling học phân biệt cặp xuất hiện thật với một số cặp giả, tránh tính softmax trên toàn bộ vocabulary. Mô hình duy trì vector cho từ trung tâm và từ ngữ cảnh.

Code mẫu huấn luyện trên corpus đồ chơi; `dog` là hàng xóm gần nhất của `cat` với cosine `0.988`. Notebook thực hành tự viết một vòng huấn luyện Skip-gram với vector 8 chiều, 192 positive pairs và seed cố định. Kết quả toy run cho cosine `coffee–tea = 0.993`, cao hơn `coffee–lemon = 0.795`. Thí nghiệm window cho thấy tăng cửa sổ từ 1 lên 2 và 3 làm số cặp tăng lần lượt từ 10 lên 18 và 24.

Những con số trên chỉ mô tả corpus và seed nhỏ này, không phải kết luận ngôn ngữ tổng quát. Word2Vec cũng gán một vector cho mỗi từ, nên không phân biệt các nghĩa khác nhau của một từ như `bank`; từ chưa xuất hiện lúc huấn luyện cũng không có vector.

### 4. GloVe, FastText và Subword Embeddings

[Notebook Bài 04](../phases/05-nlp-foundations-to-advanced/04-glove-fasttext-subword/notebook/04-glove-fasttext-subword.ipynb) · [Tài liệu](../phases/05-nlp-foundations-to-advanced/04-glove-fasttext-subword/docs/en.md)

GloVe học từ thống kê đồng xuất hiện toàn cục và factorization có weighting. FastText biểu diễn từ bằng các character n-gram, giúp ghép tín hiệu từ các mảnh quen thuộc khi gặp từ hiếm hoặc chưa thấy. BPE học vocabulary subword bằng cách liên tục gộp cặp liền kề xuất hiện nhiều. Khi dùng transformer, tokenizer phải khớp model đã huấn luyện.

Code mẫu tìm thấy 10 n-gram chung giữa `where` và `whereupon`, sau đó học 10 BPE merges trên corpus nhỏ. Notebook thực hành biểu diễn từ bằng vector đếm n-gram: cosine `where–whereupon = 0.530`, còn `where–dog = 0.000`. Phần BPE in các merge và segmentation cho cả từ trong corpus lẫn từ mới như `newish`.

Vector n-gram trong bài thực hành là bag-of-substrings để quan sát phần ký tự chung, không phải FastText đã huấn luyện. Corpus BPE rất nhỏ nên segmentation chỉ dùng để hiểu thuật toán, không nên dùng làm tokenizer cho model thật.

## Mạch kiến thức của bốn bài

1. Tokenizer quyết định đơn vị đầu vào.
2. BoW/TF-IDF đếm các đơn vị đó và tạo baseline thưa, dễ giải thích.
3. Word2Vec thay phép đếm độc lập bằng vector học từ ngữ cảnh.
4. GloVe khai thác đồng xuất hiện toàn cục; FastText/BPE dùng cấu trúc subword để xử lý từ hiếm và từ chưa thấy.

Đây là một chuỗi lựa chọn biểu diễn, không phải tiến trình mà phương pháp mới luôn thay thế hoàn toàn phương pháp cũ. TF-IDF vẫn phù hợp khi tín hiệu nằm ở từ khóa; embedding hữu ích khi cần khái quát theo ngữ cảnh; subword giúp giảm vấn đề từ ngoài vocabulary.

## Thực hành đã hoàn thành

| Bài | Phần thực hành trong notebook | Output tiêu biểu |
|---|---|---|
| 01 | Tokenizer, luật stem, bảng lemma và POS toy trên ba câu | `ponies → poni` (stem), `ponies → pony` (lemma) |
| 02 | Tạo vocabulary/BoW/TF-IDF, chuẩn hóa, cosine và xếp hạng ticket | Ticket refund bị trễ đứng đầu, cosine `0.738` |
| 03 | Skip-gram pairs, negative sampling, huấn luyện NumPy và thử cửa sổ | `192` cặp; window `1/2/3` cho `10/18/24` cặp trong probe |
| 04 | Đếm character n-gram và học/áp dụng BPE merges | `where–whereupon` cosine `0.530`; in segmentation cho từ mới |

## Kết quả chạy

- Cả bốn `code/main.py` gốc đều được chạy từ cell trong notebook và ghi `Exit code: 0`.
- Tất cả code cell thực hành đều đã chạy và output được lưu trong notebook.
- Bốn notebook không có cell lỗi sau lần chạy cuối.
- Không chụp ảnh theo yêu cầu; người học có thể mở notebook và tự chụp các output cần đưa vào báo cáo nộp.

## Cách mở lại

Từ thư mục gốc repo, mở notebook đầu tiên bằng:

```powershell
.\.venv\Scripts\python.exe -m jupyter lab phases\05-nlp-foundations-to-advanced\01-text-processing\notebook\01-text-processing.ipynb
```

Trong Jupyter, chọn **Kernel → Restart Kernel and Run All Cells** để tự chạy lại từ đầu. Các notebook còn lại nằm trong thư mục `notebook/` của từng bài tương ứng.
