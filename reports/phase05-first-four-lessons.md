# Phase 05 — Báo cáo học tập: lý thuyết chi tiết 4 bài đầu

**Phạm vi:** Bài 01–04 · Text Processing, Bag of Words/TF-IDF, Word2Vec, GloVe/FastText/Subword  
**Môi trường chạy:** Python 3.12.10 trong môi trường .venv của repo  
**Cách học:** mỗi notebook có phần lý thuyết, chạy code mẫu gốc, code thực hành tự viết và câu hỏi tự ôn. Output đã lưu trong notebook; không tạo ảnh chụp.

> Phase 05 có 29 bài. Báo cáo này giới hạn ở bốn bài đầu theo yêu cầu hiện tại.

## 1. Text Processing — Tokenization, Stemming, Lemmatization

[Notebook Bài 01](../phases/05-nlp-foundations-to-advanced/01-text-processing/notebook/01-text-processing.ipynb) · [Tài liệu gốc](../phases/05-nlp-foundations-to-advanced/01-text-processing/docs/en.md)

### Lý thuyết

Một pipeline NLP cổ điển không đọc trực tiếp câu như con người. Nó phải xác định đơn vị văn bản, gán biểu diễn cho đơn vị đó và áp dụng cùng một quy ước khi huấn luyện lẫn dự đoán. Tokenization là quyết định ranh giới: từ, subword, ký tự hay byte. Chữ hoa/thường, dấu câu, apostrophe, số, URL và emoji đều có thể làm thay đổi kết quả tách. Không có tokenizer “đúng cho mọi tác vụ”; quyết định phù hợp phụ thuộc ngôn ngữ và downstream task.

Stemming dùng luật cắt hậu tố để gom một số biến thể về gần cùng dạng. Code bài triển khai Porter step 1a: xét các hậu tố theo thứ tự, ví dụ “sses” và “ies” trước, giữ “ss”, rồi mới bỏ “s” đơn. Vì vậy “caresses” thành “caress”, “ponies” thành “poni”. Stem có thể không phải từ trong từ điển. Nó nhanh và hữu ích cho tìm kiếm thô, nhưng có thể over-stem (gộp nhầm hai từ) hoặc under-stem (không gom đủ biến thể).

Lemmatization tìm dạng từ điển dựa trên hình thái và thường cần POS. “running → run”, “ran → run”, “were → be” đòi hỏi biết vai trò/ngữ pháp của từ; chỉ cắt chuỗi không thể xử lý đáng tin cậy các dạng bất quy tắc. Code bài dùng lookup nhỏ khóa bằng (word, POS), rồi fallback cho “-ing” và số nhiều danh từ. Điều này giải thích cơ chế nhưng chưa phải lemmatizer đầy đủ: “watched” không đổi và POS tagger toy gán phần lớn token thành NOUN.

Các bước preprocessing cần được đóng gói và dùng thống nhất cho train/inference. Nếu một phía lowercase, stem hoặc bỏ stopword mà phía kia không làm, vocabulary/feature distribution đổi và mô hình có thể suy giảm âm thầm. Cũng cần thận trọng khi áp dụng regex tiếng Anh cho tiếng Việt: code bài giới hạn chữ cái ở A–Z, trong khi tiếng Việt có Unicode dấu và khoảng trắng không luôn tương đương ranh giới từ.

### Thực thi và thực hành

Code mẫu chạy câu “The cats were running at 3pm.”: tokenizer tách “3pm” thành “3” và “pm”; lemma chuyển “running” thành “run”, “were” thành “be”. Trong notebook, ba phép biến đổi được so trên câu mới: stem “ponies” thành “poni”, lookup lemma thành “pony”; “watched” được giữ nguyên do bảng toy chưa bao phủ “-ed”. Ví dụ này làm rõ rằng cùng một từ có thể có token gốc, stem và lemma khác nhau.

**Kết luận bài:** chọn preprocessing theo tác vụ, đo ảnh hưởng của nó và giữ pipeline nhất quán. Đừng coi stemming và lemmatization là hai tên gọi cho cùng một thao tác.

## 2. Bag of Words, TF-IDF và Text Representation

[Notebook Bài 02](../phases/05-nlp-foundations-to-advanced/02-bag-of-words-tfidf/notebook/02-bag-of-words-tfidf.ipynb) · [Tài liệu gốc](../phases/05-nlp-foundations-to-advanced/02-bag-of-words-tfidf/docs/en.md)

### Lý thuyết

Với N tài liệu và vocabulary có V token, BoW tạo ma trận N × V. Ô X[i,j] đếm token j trong tài liệu i. Mọi hàng có cùng số chiều; phần lớn ô thường bằng 0 vì tài liệu chỉ chứa một phần vocabulary. Ma trận này dễ diễn giải nhưng bỏ thứ tự: “dog bites man” và “man bites dog” có cùng vector unigram nếu tần suất từ giống nhau.

Term Frequency (TF) đo mức độ từ xuất hiện trong một tài liệu. Bài chuẩn hóa count theo độ dài tài liệu để tài liệu dài không tự động có trọng số cao chỉ vì chứa nhiều token. Document Frequency df đếm số tài liệu có chứa từ đó. IDF giảm trọng số cho từ xuất hiện rộng khắp và nhấn từ đặc trưng. Code dùng công thức có smoothing:

TF-IDF(w,d) = TF(w,d) × [log((N + 1)/(df(w) + 1)) + 1]

Cộng 1 trong tử/mẫu tránh chia cho 0; cộng 1 bên ngoài giữ trọng số dương cho từ xuất hiện trong mọi tài liệu. Các thư viện có thể dùng biến thể khác, nên các con số không phải lúc nào cũng khớp chính xác.

Sau đó L2 normalization chia hàng cho sqrt(sum(x²)). Hai hàng chuẩn hóa được so bằng cosine similarity; khi norm bằng 1, cosine rút thành dot product. Score cao nghĩa là hai văn bản gần nhau trong không gian feature này, không chứng minh chúng đồng nghĩa. BoW/TF-IDF nhanh, thưa, rẻ và giải thích được; nó không biết rằng hai từ khác nhau có thể đồng nghĩa hoặc rằng phủ định đảo nghĩa.

N-gram có thể thêm cụm như “not_good” để giữ một phần thứ tự. Đổi lại vocabulary lớn và thưa hơn; n-gram vẫn không nắm cấu trúc câu tổng quát. Với test/inference, vocabulary và IDF cần fit trên corpus huấn luyện rồi tái sử dụng; từ mới ngoài vocabulary không tự có một feature đã học.

### Thực thi và thực hành

Code mẫu tạo ba câu, vocabulary và ma trận cosine. Hai câu cùng “cat/sat/on/mat” có similarity 0.80; cặp ít từ chung hơn có 0.46. Notebook thực hành tạo vector TF-IDF ticket và xếp query “refund delayed bank transfer”: ticket refund bị trễ đứng đầu với cosine 0.738. Thí nghiệm bigram cho thấy “not_good” xuất hiện ở review phủ định.

**Kết luận bài:** TF-IDF là baseline tốt khi từ khóa mang tín hiệu và cần tốc độ/khả năng giải thích. Khi ngữ nghĩa, thứ tự hoặc từ đồng nghĩa quan trọng, cần thêm n-gram, embedding hoặc mô hình theo ngữ cảnh.

## 3. Word Embeddings — Word2Vec

[Notebook Bài 03](../phases/05-nlp-foundations-to-advanced/03-word-embeddings-word2vec/notebook/03-word-embeddings-word2vec.ipynb) · [Tài liệu gốc](../phases/05-nlp-foundations-to-advanced/03-word-embeddings-word2vec/docs/en.md)

### Lý thuyết

Distributional hypothesis nói rằng từ xuất hiện trong ngữ cảnh tương tự thường có nghĩa gần nhau. Word2Vec học vector d chiều thay vì dành một chiều riêng cho mỗi từ như BoW. Tín hiệu học đến từ các cửa sổ từ trong corpus, không phải nhãn nghĩa viết tay; vì vậy vector phản ánh corpus, preprocessing, kích thước window và seed.

Skip-gram lấy một từ trung tâm để dự đoán các từ lân cận. Với cửa sổ hẹp, cặp context chủ yếu là cục bộ; cửa sổ rộng thêm cặp xa và làm thay đổi mục tiêu. CBOW làm ngược lại: gộp context để dự đoán center. Trong Word2Vec có bảng W cho center vectors và W′ cho context vectors. One-hot của center tương đương chọn một hàng; tích vô hướng giữa vector center/context tạo score cho cặp.

Softmax đầy đủ phải chuẩn hóa xác suất trên toàn vocabulary, chi phí lớn khi vocabulary nhiều token. Negative sampling thay bài toán đó bằng một cặp thật và vài cặp giả. Với vector center v_c, context thật u_o và negative vectors u_k, một dạng loss là:

L = −log σ(v_c · u_o) − Σ_k log σ(−v_c · u_k)

Positive term kéo score của cặp quan sát được lên; negative terms đẩy score các cặp mẫu giả xuống. Gradient cập nhật cả bảng center và context. Phân phối chọn negative và số lượng negative samples ảnh hưởng đến học; notebook dùng hai negative samples lấy đều để code ngắn, trong khi cấu hình Word2Vec thường có điều chỉnh theo tần suất.

Sau huấn luyện, cosine thường dùng tìm từ gần nhau. Nhưng cosine cao không tự chứng minh đồng nghĩa; từ trái nghĩa cũng có thể xuất hiện trong ngữ cảnh tương tự. Word2Vec là static embedding, một vector cho mỗi loại từ nên không tách được “bank” ở bờ sông và ngân hàng. Từ OOV không có vector mặc định. Contextual embeddings và FastText xử lý hai hạn chế này theo những cách khác nhau.

### Thực thi và thực hành

Code mẫu huấn luyện toy corpus, trong đó “dog” là hàng xóm gần “cat” với cosine 0.988. Notebook tự tạo 192 cặp Skip-gram, huấn luyện vector 8 chiều trên corpus đồ uống; kết quả seed cố định có cosine coffee–tea 0.993 và coffee–lemon 0.795. Probe dài 6 token cho thấy window 1/2/3 tạo 10/18/24 cặp.

**Kết luận bài:** Word2Vec học hình học từ ngữ cảnh; kết quả phải được diễn giải cùng corpus và cấu hình, không xem toy similarity là đánh giá ngữ nghĩa phổ quát.

## 4. GloVe, FastText và Subword Embeddings

[Notebook Bài 04](../phases/05-nlp-foundations-to-advanced/04-glove-fasttext-subword/notebook/04-glove-fasttext-subword.ipynb) · [Tài liệu gốc](../phases/05-nlp-foundations-to-advanced/04-glove-fasttext-subword/docs/en.md)

### Lý thuyết

GloVe bắt đầu từ ma trận đồng xuất hiện X, trong đó X[i,j] là số lần j nằm trong context của i. Mục tiêu là làm tích vô hướng của center/context vectors cộng bias xấp xỉ log count: w_i · w̃_j + b_i + b̃_j ≈ log(X[i,j]). Loss có hàm weighting để cặp rất hiếm hoặc rất phổ biến không lấn át mọi cặp khác. GloVe học từ thống kê tổng hợp toàn corpus; Skip-gram học từ các cặp cửa sổ trong quá trình dự đoán.

FastText thêm cấu trúc ký tự. Từ được bọc ranh giới, chia thành character n-gram, rồi vector từ được tạo từ vector các mảnh và thường cả vector từ nguyên vẹn. Từ OOV có thể vẫn biểu diễn được nếu nó chia sẻ n-gram đã học với từ khác. Cách này hữu ích với biến tố và từ hiếm, nhưng mảnh ký tự giống nhau không đảm bảo nghĩa giống nhau.

BPE giải quyết vấn đề vocabulary/tokenization theo hướng khác: nó học merge giữa các token liền kề. Bắt đầu từ ký tự/byte, đếm cặp (tính cả tần suất của từ), gộp cặp phổ biến nhất, lặp đến số merge định trước. Khi áp dụng, phải replay merges theo cùng thứ tự. BPE cho ra chuỗi subword token ID; nó không tự học vector ngữ nghĩa. Model transformer học embedding cho các token đó. Byte-level encoding giúp bao phủ chuỗi lạ, nhưng từ hiếm có thể tách thành nhiều token. Tokenizer là một phần của model đã train, không thể thay tùy ý lúc inference.

### So sánh vai trò

| Phương pháp | Tín hiệu | Đầu ra | Giải quyết tốt | Hạn chế |
|---|---|---|---|---|
| GloVe | Đồng xuất hiện toàn corpus | Vector tĩnh theo từ | Thống kê toàn cục | Cần cách xử lý OOV |
| FastText | Word context cùng character n-gram | Vector từ tổng hợp từ mảnh | Từ hiếm/OOV có mảnh quen thuộc | Mảnh ký tự không bảo đảm ngữ nghĩa |
| BPE | Tần suất cặp token/byte | Chuỗi subword ID | Tokenizer bao phủ chuỗi chưa thấy | Vocabulary/độ dài chuỗi là tradeoff |

### Thực thi và thực hành

Code mẫu đếm 10 n-gram chung giữa “where” và “whereupon”, rồi học 10 BPE merges từ corpus nhỏ. Notebook so vector đếm n-gram toy: cosine where–whereupon là 0.530, where–dog là 0.000; tiếp đó học và áp dụng merges cho từ trong/ngoài corpus. Phần n-gram này chỉ minh họa phần ký tự chung, không phải FastText đã huấn luyện; corpus BPE nhỏ cũng chỉ để học thuật toán.

**Kết luận bài:** GloVe và Word2Vec tạo vector tĩnh bằng tín hiệu phân bố khác nhau; FastText tạo vector có cấu trúc subword; BPE quyết định đơn vị token mà model xử lý. Cần chọn đúng loại biểu diễn theo nhiệm vụ, không gộp chúng thành một thuật toán chung.

## Mạch kiến thức của bốn bài

1. Tokenization xác định đơn vị đầu vào.
2. BoW/TF-IDF biến các đơn vị đó thành vector đếm thưa, dễ giải thích.
3. Word2Vec học vector từ ngữ cảnh lân cận.
4. GloVe khai thác đồng xuất hiện toàn cục; FastText tận dụng mảnh ký tự; BPE học chuỗi subword cho tokenizer.

Các phương pháp mới không xóa bỏ hoàn toàn phương pháp cũ. TF-IDF vẫn hữu ích cho từ khóa, latency thấp và dữ liệu ít; embedding giúp khái quát theo ngữ cảnh; subword giảm vấn đề từ lạ. Quyết định cần dựa trên dữ liệu và tiêu chí đánh giá.

## Kết quả thực hành

| Bài | Phần tự viết | Output tiêu biểu |
|---|---|---|
| 01 | Tokenizer, luật stem, lookup lemma và POS toy | ponies → poni (stem), ponies → pony (lemma) |
| 02 | Vocabulary, BoW/TF-IDF, chuẩn hóa, cosine và xếp hạng | Ticket refund trễ đứng đầu, cosine 0.738 |
| 03 | Skip-gram pairs, negative sampling, huấn luyện NumPy, thay window | 192 cặp; window 1/2/3 tạo 10/18/24 cặp trong probe |
| 04 | Character n-gram và học/áp dụng BPE | where–whereupon cosine 0.530; có segmentation từ mới |

## Trạng thái chạy

- Cả bốn code/main.py gốc chạy trong notebook và ghi Exit code: 0.
- Các cell thực hành đều đã chạy; output đã lưu trong notebook.
- Không có cell lỗi ở lần chạy cuối.
- Không tạo ảnh; người học tự chụp output trong notebook nếu cần nộp ảnh minh chứng.
