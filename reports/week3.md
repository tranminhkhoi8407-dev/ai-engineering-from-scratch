# Phase 05 — Báo cáo học tập tuần 3: Bài 05–08

**Phạm vi:** Bài 05–08 · Sentiment Analysis, Named Entity Recognition, POS Tagging & Syntactic Parsing, CNNs & RNNs for Text
**Môi trường chạy:** Python 3.12.10, NumPy 2.5.4, Jupyter nbconvert 7.17.1 (kernel `python3`)
**Cách làm:** mỗi bài có một notebook 8 cell trong thư mục `notebook/` của bài: cell lý thuyết, phần A chạy `code/main.py` gốc qua `subprocess`, phần B/C tự viết lại cơ chế và đo kết quả, phần D tự kiểm tra. Toàn bộ output đã lưu trong notebook; không tạo ảnh chụp.

> Phase 05 có 29 bài. Báo cáo này giới hạn ở bốn bài 05–08 theo yêu cầu hiện tại.

## Bảng tổng quan

| Bài | Chủ đề | Trọng tâm thực hành | Kết quả nổi bật |
| --- | --- | --- | --- |
| 05 | Sentiment Analysis | Naive Bayes, 4 biến thể đặc trưng, probe phủ định, logistic regression, macro-F1 | Biến thể unigram đạt accuracy 0.800; probe phủ định 0.500 → 1.000 khi bật scoping |
| 06 | Named Entity Recognition | BIO/BILOU, gazetteer, entity-level F1, Viterbi có ràng buộc, nested entity | Token accuracy 0.875 nhưng entity micro-F1 0.750; Viterbi sửa chuỗi `O → I-GPE` |
| 07 | POS Tagging & Parsing | MFT, bigram HMM, Viterbi, dependency tree, SVO, PP-attachment | MFT 0.938, HMM 1.000 trên 32 token; HMM vẫn sai ở câu OOV của code mẫu |
| 08 | CNNs & RNNs for Text | Conv1D + max-pool, bất biến vị trí, LSTM gates, vanishing gradient | Gradient RNN còn 2.66e−5 sau 100 bước; LSTM với f=0.99 còn 0.366 |

## 1. Bài 05 — Sentiment Analysis

[Notebook Bài 05](../phases/05-nlp-foundations-to-advanced/05-sentiment-analysis/notebook/05-sentiment-analysis.ipynb) · [Tài liệu gốc](../phases/05-nlp-foundations-to-advanced/05-sentiment-analysis/docs/en.md)

### 1.1 Lý thuyết

Sentiment là bài phân loại văn bản hai bước: **biểu diễn** văn bản thành vector đặc trưng, rồi **phân loại** bằng mô hình tuyến tính. Mọi khó khăn của NLP cổ điển xuất hiện ở đây dưới dạng nén: phủ định, mỉa mai, từ đa nghĩa theo miền, emoji, và dữ liệu lệch lớp.

Multinomial Naive Bayes giả định các đặc trưng độc lập có điều kiện theo nhãn:

P(y | x) ∝ P(y) · Π P(x_i | y)

với xác suất từ được ước lượng bằng đếm cộng làm trơn Laplace:

P(w | y) = (count(w, y) + α) / (Σ_v count(v, y) + α·|V|)

Giả định độc lập sai nhưng kết quả vẫn mạnh trên đặc trưng thưa, vì mô hình chỉ cần biết mỗi từ nghiêng về phía nào, không cần ước lượng chính xác phân phối đồng thời. Hệ quả ít được nói: NB cộng log xác suất của mọi token, nên độ dài tài liệu và tổng khối lượng token của mỗi lớp ảnh hưởng trực tiếp đến quyết định.

Logistic regression bỏ giả định độc lập, học trọng số thực cho từng đặc trưng:

P(y = 1 | x) = σ(w · x + b),  L = −(1/N) Σ [y log p + (1 − y) log(1 − p)] + λ‖w‖²

Trọng số âm là thứ NB không biểu diễn tự nhiên. Với bigram `NOT_good`, logistic regression có thể học một trọng số âm mạnh.

Negation scoping gắn tiền tố `NOT_` cho token sau từ phủ định cho đến dấu câu gần nhất. Đây là quy ước thô: `not only good but excellent` bị xử lý sai, và phủ định phạm vi xa cũng khó. Scoping không tạo thông tin mới; nó chỉ tách một đặc trưng thành hai, nên chỉ có tác dụng nếu dữ liệu huấn luyện có ví dụ `NOT_x` ở cả hai lớp.

Về đo lường, accuracy vô dụng khi lớp lệch. Cần precision/recall/F1 từng lớp, macro-F1, confusion matrix và mẫu lỗi đọc bằng mắt. Với lệch trên 95–5, dùng AUROC/AUPRC. Transformer chỉ thực sự cần cho mỉa mai, review dài đổi cảm xúc, aspect-based sentiment và ngôn ngữ ít tài nguyên.

### 1.2 Chạy code mẫu

`code/main.py` huấn luyện NB trên 5 review positive + 5 review negative, có negation scoping, rồi dự đoán 5 câu test. Kết quả thực tế: 3 đúng, 2 sai.

| Câu test | Nhãn thật | Dự đoán |
| --- | --- | --- |
| this movie was not good | − | + (sai) |
| loved every minute | + | + |
| terrible waste of time | − | + (sai) |
| beautiful and brilliant | + | + |
| i did not enjoy this at all | − | − |

Metric in ra: `tp=2, fp=2, tn=1, fn=0`, precision 0.500, recall 1.000, F1 0.667. Recall 1.0 nhưng precision 0.5: mô hình không bỏ sót câu negative nào nhưng đoán positive quá rộng.

Hai lỗi có nguyên nhân cụ thể, không phải "mô hình chưa đủ thông minh":

- `this movie was not good` chứa `NOT_good`, nhưng `NOT_good` không xuất hiện trong tập train (train chỉ có `NOT_bad`, `NOT_worth`, `NOT_recommend`…). Không có bằng chứng cho `NOT_good`, quyết định rơi về các token chức năng và prior.
- `terrible waste of time`: từ `terrible` có trong train negative, nhưng điểm của một stopword như `of` lại lớn hơn. Với 5 tài liệu ngắn mỗi lớp, một token phổ biến có thể lấn át đúng từ mang cảm xúc. Đây là hiệu ứng tần suất/độ dài, không phải lỗi logic.

### 1.3 Thực hành trong notebook

Notebook tự viết lại pipeline và so bốn biến thể đặc trưng trên cùng 12 review train, 10 review test:

| Biến thể | Accuracy | Macro-F1 |
| --- | --- | --- |
| unigram | 0.800 | 0.800 |
| unigram + negation | 0.800 | 0.800 |
| unigram + bigram | 0.700 | 0.697 |
| unigram + bigram + negation | 0.700 | 0.697 |

Trên corpus nhỏ này, thêm bigram **không** giúp: số tham số tăng, tần suất mỗi đặc trưng giảm, ước lượng nhiễu hơn. Đây là bài học thực nghiệm đáng giữ: nhiều đặc trưng hơn không đồng nghĩa mô hình tốt hơn khi dữ liệu ít.

Vì bảng trên trộn nhiều yếu tố, notebook thêm một **probe có kiểm soát**: cùng một mô hình, cùng dạng câu, chỉ đổi từ phủ định.

| Mô hình | Accuracy trên 4 câu probe | Dự đoán |
| --- | --- | --- |
| unigram | 0.500 | `[+, +, +, +]` |
| unigram + negation | 1.000 | `[+, −, −, +]` |

Mô hình không scoping gán **cùng một nhãn** cho `the room was good` và `the room was not good`, và cũng gán cùng nhãn cho cặp `bad` / `not bad`. Đây là failure mode được cô lập, không phải suy đoán từ accuracy tổng.

Logistic regression trên đặc trưng `unigram + bigram + negation` cho thấy mô hình học đúng hướng:

| Đặc trưng | Trọng số |
| --- | --- |
| `great` | +0.586 |
| `,` | +0.468 |
| `cinematography` | +0.351 |
| `NOT_bad` | +0.282 |
| `NOT_worth` | −0.277 |
| `NOT_acting` | −0.193 |
| `boring`, `far_too`, `and_far` | −0.406 |

Đáng chú ý: hai đặc trưng cùng tiền tố `NOT_` nhận trọng số ngược dấu. `NOT_bad` dương vì nó xuất hiện trong câu positive; `NOT_worth` âm vì xuất hiện trong câu negative. Đó chính là điều NB khó biểu diễn và là lý do dùng logistic regression khi cần trọng số có dấu.

Một artifact của scoping lộ ra khi đọc danh sách đặc trưng: `NOT_acting` âm là do câu `the plot made no sense and the acting was terrible` bị `no` scope đến hết câu, biến `terrible` thành `NOT_terrible`. Scoping theo dấu câu không phân biệt phủ định cục bộ với phủ định toàn mệnh đề.

Cuối cùng, thí nghiệm lệch lớp: bộ test 9 positive + 1 negative, mô hình luôn đoán positive đạt accuracy 0.900 nhưng macro-F1 chỉ 0.474; F1 lớp positive 0.947, F1 lớp negative 0.000. Con số này minh họa trực tiếp vì sao accuracy một mình là metric nguy hiểm.

### 1.4 Kết luận bài

Sentiment cổ điển là baseline hợp lý khi tín hiệu nằm ở từ khóa và cần tốc độ/khả năng giải thích. Nhưng kết quả trên corpus nhỏ phụ thuộc mạnh vào dữ liệu: từ phủ định chưa thấy, stopword phổ biến, độ dài lớp. Muốn kết luận về chất lượng thật phải chạy trên split chuẩn và đọc trực tiếp các lỗi.

## 2. Bài 06 — Named Entity Recognition

[Notebook Bài 06](../phases/05-nlp-foundations-to-advanced/06-named-entity-recognition/notebook/06-named-entity-recognition.ipynb) · [Tài liệu gốc](../phases/05-nlp-foundations-to-advanced/06-named-entity-recognition/docs/en.md)

### 2.1 Lý thuyết

BIO biến trích xuất thực thể thành gán nhãn chuỗi: `B-X` mở đầu, `I-X` tiếp diễn, `O` nằm ngoài. BILOU thêm `L-X` (token cuối) và `U-X` (thực thể một token), giúp ranh giới rõ hơn nhưng tăng số nhãn.

Điểm kỹ thuật quan trọng: BIO cho phép chuỗi bất hợp lệ như `O I-ORG` hoặc `B-GPE I-ORG`. Bộ giải mã argmax từng token tạo ra chúng thường xuyên. CRF học transition nên đường đi tối ưu toàn cục luôn hợp lệ.

Tiến trình kiến trúc: rule-based + gazetteer (precision cao, coverage thấp, không phân biệt Apple công ty và quả táo) → HMM (emission + transition, Viterbi) → CRF (đặc trưng tùy ý, vẫn là workhorse cổ điển) → BiLSTM-CRF (đặc trưng học được + ràng buộc chuỗi) → transformer fine-tune → LLM zero-shot/few-shot.

Đo lường phải là **entity-level F1**: span dự đoán khớp chính xác start, end và kiểu. Token accuracy bị thổi phồng vì phần lớn token là `O`; một mô hình gán mọi token `O` vẫn có thể đạt 85–90% token accuracy với entity F1 bằng 0.

Giới hạn cấu trúc: nested entity (`Bank of America Tower` vừa ORG vừa FACILITY) không biểu diễn được bằng một mảng BIO; cần multi-pass hoặc mô hình span-based. Domain shift, thực thể dài và kiểu hiếm cũng là các điểm gãy đã biết.

### 2.2 Chạy code mẫu

`code/main.py` gán nhãn câu `Apple sued Google over iPhone sales in the US .` bằng gazetteer:

- Apple, Google → ORG
- iPhone → PRODUCT
- US → GPE

Word shape in ra: `Apple → Xxxxx`, `iPhone → xXxxxx`, `IBM → XXX`, `USA-2024 → XXX-dddd`, `apple → xxxxx`. Đây là các đặc trưng mạnh cho CRF cổ điển.

Round-trip `spans → BIO → spans` trên câu `The New York City mayor visited OpenAI .` cho kết quả khớp hoàn toàn: BIO `O B-GPE I-GPE I-GPE O O B-ORG O`, thu hồi lại `[(1, 4, 'GPE'), (6, 7, 'ORG')]`, `match: True`.

### 2.3 Thực hành: token accuracy so với entity F1

Notebook dựng bộ gán nhãn theo luật: khớp cụm gazetteer trước (`New York City`, `Bank of America`), sau đó khớp token gazetteer, cuối cùng đoán `B-ORG` cho token viết hoa chưa biết (có stoplist cho từ mở câu). Chạy trên 3 câu test chứa thực thể ngoài gazetteer:

| Câu | Gold | Dự đoán |
| --- | --- | --- |
| Tesla opened a plant in Berlin . | Tesla ORG, Berlin GPE | Tesla ORG, Berlin **ORG** |
| The New York City mayor met Satya Nadella . | New York City GPE, Satya Nadella PERSON | New York City GPE, Satya Nadella **ORG** |
| Anthropic released Claude for Android in Japan . | đúng 4 thực thể | đúng 4 thực thể |

Kết quả:

| Metric | Giá trị |
| --- | --- |
| Token accuracy | 0.875 |
| Entity precision | 0.750 |
| Entity recall | 0.750 |
| Entity micro-F1 | 0.750 |
| Entity macro-F1 | 0.617 |

Per-type: ORG precision 0.500 / recall 1.000 / F1 0.667; GPE 1.000 / 0.667 / 0.800; PERSON 0 / 0 / 0; PRODUCT 1.000 / 1.000 / 1.000.

Khoảng cách 0.875 so với 0.750 là toàn bộ bài học: token accuracy chỉ ra 3 lỗi trên 24 token, trông gần hoàn hảo; entity F1 cho thấy 2 trong 8 thực thể sai, và kiểu PERSON bị bỏ hoàn toàn. Nếu báo cáo token accuracy, người đọc sẽ tưởng hệ thống đã xong.

### 2.4 Thực hành: giải mã có ràng buộc

Notebook gán điểm phát xạ cho từng token của câu `The New York City mayor` (điểm do người viết đặt để cô lập hành vi giải mã):

- argmax từng token: `O O I-GPE I-GPE O`
- Viterbi không ràng buộc: giống hệt argmax (vì transition bằng 0)
- Viterbi có ràng buộc BIO: `O B-GPE I-GPE I-GPE O`
- Vi phạm của argmax: `(2, 'O', 'I-GPE')`
- Span từ argmax: `[]`; span từ constrained: `[(1, 4, 'GPE')]`

Đây là minh chứng cơ học cho vai trò của CRF layer: bản thân điểm phát xạ không đủ; ràng buộc chuyển trạng thái biến một chuỗi vô nghĩa thành một thực thể đúng. Việc Viterbi "không ràng buộc" trùng argmax cũng chứng minh chênh lệch không đến từ thuật toán giải mã mà từ ràng buộc.

### 2.5 Nested entity và BILOU

Với `Bank of America Tower in New York City .`, gán lần lượt span ORG `(0,3)` và FAC `(0,4)` vào cùng một mảng nhãn:

- BIO một lượt: `B-FAC I-FAC I-FAC I-FAC O …`, thu hồi chỉ còn `[(0, 4, 'FAC')]`
- Lượt ORG riêng: `B-ORG I-ORG I-ORG O …` → `[(0, 3, 'ORG')]`
- Lượt FAC riêng: `B-FAC I-FAC I-FAC I-FAC O …` → `[(0, 4, 'FAC')]`

Span sau ghi đè span trước, nên ORG **mất thông tin**, không chỉ bị nén. Multi-pass theo kiểu giữ được cả hai vì mỗi lượt là bài toán nhị phân độc lập. Đây là cách các pipeline hợp đồng/y tế xử lý ontology lồng nhau.

BILOU trên `The New York City mayor visited Acme .`:

```text
BIO   : O B-GPE I-GPE I-GPE O B-GPE O B-ORG
BILOU : O B-GPE I-GPE L-GPE O U-GPE O U-ORG
```

Round-trip BILOU khớp gold. `U-GPE` mã hóa thực thể một token, `L-GPE` đánh dấu kết thúc — thông tin mà BIO không có.

### 2.6 Kết luận bài

Chọn metric đúng quan trọng ngang chọn mô hình. Với NER, entity-level F1 là bắt buộc; token accuracy chỉ dùng để chẩn đoán. Ràng buộc chuỗi là thành phần kiến trúc, không phải post-process tùy chọn. Và BIO có trần biểu diễn: nested entity cần thiết kế khác ngay từ đầu.

## 3. Bài 07 — POS Tagging và Syntactic Parsing

[Notebook Bài 07](../phases/05-nlp-foundations-to-advanced/07-pos-tagging-parsing/notebook/07-pos-tagging-parsing.ipynb) · [Tài liệu gốc](../phases/05-nlp-foundations-to-advanced/07-pos-tagging-parsing/docs/en.md)

### 3.1 Lý thuyết

POS tagging gán phạm trù ngữ pháp cho token. Penn Treebank có 36 tag tiếng Anh, phân biệt thời thì và số; Universal Dependencies có 17 tag, ngôn ngữ trung lập và là mặc định cho cross-lingual.

Baseline most-frequent-tag (MFT) chọn tag phổ biến nhất của mỗi từ trong training, đạt khoảng 85% trên Brown. Nó thất bại ở từ mơ hồ: `book` trong `the book` là NOUN, trong `they book flights` là VERB.

Bigram HMM mô hình hóa:

P(tags, words) = Π P(tag_i | tag_{i-1}) · P(word_i | tag_i)

và giải mã bằng Viterbi:

V[i][j] = emission(i, j) + max_k ( V[i−1][k] + transition(k, j) )

Chi phí O(n·|T|²), |T| khoảng 17–36 nên rất nhỏ. Trên Brown, HMM đạt khoảng 93%; phần lớn bước nhảy 85 → 93 đến từ transition: mô hình học `DET NOUN` phổ biến, `NOUN DET` hiếm. Nhờ vậy một từ OOV sau `DET` vẫn được gán NOUN dù emission không có thông tin.

Hạn chế: emission và transition đều cục bộ, không phân biệt `saw` trong hai ngữ cảnh nếu tập từ lân cận giống nhau. CRF với đặc trưng tùy ý đạt khoảng 97%, BiLSTM-CRF/transformer 98%+. Trần thực tế do bất đồng giữa người gán nhãn: khoảng 97% trên Penn Treebank, nên mô hình trên 98% có thể đang overfit test set.

Parsing có hai họ: transition-based (shift/reduce, nhanh, MaltParser, Chen–Manning) và graph-based (chấm điểm mọi cạnh, maximum spanning tree, chậm hơn nhưng chính xác hơn). Dependency parsing thắng trong thập niên 2010 vì tổng quát hóa sạch sang ngôn ngữ trật tự tự do. Đo bằng UAS/LAS; bẫy kinh điển là mơ hồ noun–verb và PP-attachment.

### 3.2 Chạy code mẫu

`code/main.py` huấn luyện MFT và bigram HMM trên 6 câu rồi gán nhãn 3 câu test:

| Câu | MFT | HMM |
| --- | --- | --- |
| The cat chased the dog | DET NOUN **NOUN** DET NOUN | DET NOUN **AUX** DET NOUN |
| A red mat is here | DET ADJ NOUN AUX **NOUN** | DET ADJ NOUN AUX **ADJ** |
| Dogs bark | NOUN VERB | NOUN VERB |

Đáng chú ý: HMM của code mẫu **không** tốt hơn MFT ở đây. `chased` bị gán AUX, `here` bị gán ADJ. Nguyên nhân là làm trơn cộng alpha trên từ OOV: xác suất phát xạ của một từ chưa thấy là α / (tổng token của tag + α·(|V|+1)). Tag có ít token (AUX chỉ có `is`) cho xác suất cao hơn tag có nhiều token (VERB). Với `here`, transition `AUX → ADJ` (từ `is red`) kéo nó về ADJ.

Đây là bài học quan trọng: HMM không tự động tốt hơn. Lợi thế của nó phụ thuộc vào việc transition có bù được emission hay không; với từ OOV và tag hiếm, làm trơn có thể tạo ra thiên lệch ngược.

### 3.3 Thực hành: đo accuracy và tìm chỗ HMM thắng

Notebook huấn luyện trên 12 câu tag, test trên 6 câu giữ riêng, trong đó có hai từ OOV `zorbat` đứng ở vị trí động từ:

| Câu | MFT | HMM |
| --- | --- | --- |
| The cat sat on the mat . | 7/7 | 7/7 |
| Cats zorbat mice . | 3/4 | 4/4 |
| The old man walks slowly . | 6/6 | 6/6 |
| We bought a new car . | 6/6 | 6/6 |
| She reads books quickly . | 5/5 | 5/5 |
| Dogs zorbat loudly . | 3/4 | 4/4 |

MFT accuracy 0.938, HMM 1.000 trên 32 token. Wrapper dùng danh sách tag đã sắp xếp để kết quả tất định (thứ tự `set` phụ thuộc hash seed).

Transition học được:

```text
<BOS> → DET 0.50, PRON 0.33, NOUN 0.17
DET   → NOUN 0.70, ADJ 0.30
NOUN  → VERB 0.41, PUNCT 0.35, ADV 0.12
```

`NOUN → VERB 0.41` là lý do `zorbat` sau `Cats`/`Dogs` được gán VERB dù emission hoàn toàn trống. Đây đúng là cơ chế đã đưa HMM từ 85% lên 93% trên Brown, và cũng giải thích vì sao kết quả phụ thuộc corpus: nếu training không có mẫu `NOUN VERB`, transition không cứu được.

### 3.4 Thực hành: đọc dependency tree và trích triple

Notebook dựng tay một dependency tree cho `The company bought a small startup in Berlin .`:

| i | token | pos | dep | head |
| --- | --- | --- | --- | --- |
| 0 | The | DET | det | company |
| 1 | company | NOUN | nsubj | bought |
| 2 | bought | VERB | root | ROOT |
| 3 | a | DET | det | startup |
| 4 | small | ADJ | amod | startup |
| 5 | startup | NOUN | dobj | bought |
| 6 | in | ADP | prep | bought |
| 7 | Berlin | NOUN | pobj | in |
| 8 | . | PUNCT | punct | bought |

Extractor đọc `nsubj` và `dobj` của động từ, trả về `('company', 'bought', 'startup')`. Cụm `in Berlin` bổ nghĩa cho `bought`, không phải `startup` — thông tin này nằm ở cạnh `prep`, không nằm ở token.

Câu bị động `The startup was bought by the company .` cho kết quả `(None, 'bought', None)`: extractor ngây thơ chỉ đọc `nsubj`/`dobj` nên bỏ mất chủ thể ngữ nghĩa nằm ở `agent` + `pobj`. Đây là failure mode phải xử lý riêng, không phải bug nhỏ.

PP-attachment với `I saw the man with the telescope .`:

| Parse | Head của `with` | Diễn giải |
| --- | --- | --- |
| A | `saw` | dùng kính để nhìn |
| B | `man` | người đàn ông có kính |

Cả hai hợp lệ ngữ pháp; chỉ ngữ nghĩa phân xử. Bất kỳ pipeline trích xuất nào bỏ qua bước kiểm tra này sẽ tạo ra triple sai một cách im lặng.

Cuối cùng, lemma phụ thuộc POS:

```text
'running' + VERB → 'run'
'running' + NOUN → 'running'
```

Đây là lời giải thích đầy đủ cho lời hứa ở bài 01: không có POS, lemmatizer không thể chọn đúng.

### 3.5 Kết luận bài

POS và dependency parsing vẫn là hạ tầng của trích xuất có cấu trúc. MFT là sàn, HMM là bước tiến nhờ transition, CRF/transformer vượt lên nhờ đặc trưng và ngữ cảnh. Nhưng "mô hình phức tạp hơn" không bảo đảm tốt hơn trên dữ liệu nhỏ: code mẫu cho thấy HMM có thể sai nhiều hơn MFT khi gặp OOV và làm trơn không cân xứng.

## 4. Bài 08 — CNNs và RNNs cho Text

[Notebook Bài 08](../phases/05-nlp-foundations-to-advanced/08-cnns-rnns-for-text/notebook/08-cnns-rnns-for-text.ipynb) · [Tài liệu gốc](../phases/05-nlp-foundations-to-advanced/08-cnns-rnns-for-text/docs/en.md)

### 4.1 Lý thuyết

TF-IDF và Word2Vec tạo vector phẳng, bỏ thứ tự. TextCNN và RNN lấp khoảng trống trước transformer.

**TextCNN.** Với embedding matrix L × d, filter W kích thước k × d trượt qua chuỗi:

a_i = ReLU( Σ_j Σ_c X[i+j, c] · W[j, c] + b )

Feature map có L − k + 1 phần tử; global max-pool lấy max theo thời gian. Ghép nhiều độ rộng (2, 3, 4) rồi đưa qua lớp tuyến tính. Ba tính chất: filter là bộ dò n-gram học được; max-pool bất biến vị trí; tính toán song song theo thời gian. Hạn chế: cửa sổ cố định, không thấy quan hệ xa hơn k token.

**RNN.** h_t = tanh(W x_t + U h_{t-1} + b), chia sẻ tham số qua thời gian. Backprop through time nhân gradient với U lặp lại; bán kính phổ < 1 gây vanishing, > 1 gây exploding. Với trọng số 0.9 và 100 bước, gradient còn 2.66e−5.

**LSTM.** Thêm cell state và ba cổng:

```text
f_t = σ(W_f x_t + U_f h_{t-1} + b_f)
i_t = σ(W_i x_t + U_i h_{t-1} + b_i)
o_t = σ(W_o x_t + U_o h_{t-1} + b_o)
g_t = tanh(W_g x_t + U_g h_{t-1} + b_g)
c_t = f_t ⊙ c_{t-1} + i_t ⊙ g_t
h_t = o_t ⊙ tanh(c_t)
```

Cell state cập nhật cộng, nên gradient chảy theo "đường cao tốc" chỉ bị nhân bởi cổng quên. GRU rút gọn còn hai cổng, thường tương đương với ít tham số hơn. RNN hai chiều ghép hidden state xuôi/ngược để mỗi token thấy cả hai phía — cần cho gán nhãn chuỗi.

Ba hạn chế còn lại dẫn đến attention: nút cổ chai tuần tự (1000 bước nối tiếp), context vector cố định trong encoder–decoder, và trần phụ thuộc xa. Attention giải cả ba; transformer bỏ recurrence.

CNN/RNN vẫn đúng cho edge/on-device, streaming, baseline nhanh và sequence labeling ít dữ liệu (BiLSTM-CRF).

### 4.2 Chạy code mẫu

`code/main.py` mô phỏng suy giảm gradient của RNN thuần:

| Độ dài | Hệ số còn lại (w = 0.9) |
| --- | --- |
| 10 | 3.49e−01 |
| 50 | 5.15e−03 |
| 100 | 2.66e−05 |
| 200 | 7.06e−10 |

Sau đó là một lượt convolution 1D trên 5 token × 3 chiều:

```text
width-2 filter activations: [0.69, 0.2, 0.32, 0.33]
width-3 filter activations: [0.94, 0.88, 0.62]
max-pooled features        : [0.69, 0.94]
```

Con số nhỏ nhưng đủ để thấy: filter rộng 3 tạo feature map ngắn hơn (3 vị trí so với 4), max-pool chọn activation mạnh nhất của từng filter, và vector cuối có kích thước cố định bất kể độ dài câu.

### 4.3 Thực hành: TextCNN bằng NumPy

Notebook dựng embedding 6 token × 4 chiều với các trục mang nghĩa đọc được (`not`, `good`, `bad` là các chiều riêng), rồi cài `conv1d` + ReLU + max-pool đúng công thức. Ba filter đặt tay: `not good` (w=2), `not bad` (w=2), `was not good` (w=3). Bias −1.0 buộc filter phải khớp đủ token.

Kết quả trên câu `the food was not good .`:

| Filter | Feature map | Max-pool |
| --- | --- | --- |
| `not_good(w=2)` | `[0, 0, 0, 1, 0]` | 1.0 |
| `not_bad(w=2)` | `[0, 0, 0, 0, 0]` | 0.0 |
| `was_not_good(w=3)` | `[0, 0, 1, 0]` | 1.0 |

Vector pooled cuối: `[1.0, 0.0, 1.0]`, shape `(3,)`. Filter `not_bad` không kích hoạt vì câu không có `bad` — bias âm biến filter thành phép AND của các token, chứ không chỉ khớp một token.

Bất biến vị trí:

| Thứ tự | Activations | Max |
| --- | --- | --- |
| `the food was not good .` | `[0, 0, 0, 1, 0]` | 1.00 |
| `not good the food was .` | `[1, 0, 0, 0, 0]` | 1.00 |

Cùng detector, cùng max-pool, vị trí khác nhau → cùng feature. Đây là lý do TextCNN phù hợp phân loại ("câu này có cụm phủ định không?") nhưng không phù hợp trả lời "cụm đó nằm ở đâu?".

### 4.4 Thực hành: LSTM, pooling và chi phí

Notebook chạy forward một LSTM 4 chiều trên 6 bước với trọng số cố định, in cổng và chuẩn trạng thái:

| Step | f | i | o | \|c\| | \|h\| |
| --- | --- | --- | --- | --- | --- |
| 0 | 0.760 | 0.617 | 0.449 | 0.645 | 0.256 |
| 1 | 0.878 | 0.578 | 0.448 | 0.906 | 0.375 |
| 2 | 0.817 | 0.567 | 0.531 | 1.119 | 0.539 |
| 3 | 0.805 | 0.609 | 0.529 | 1.332 | 0.628 |
| 4 | 0.826 | 0.628 | 0.444 | 1.687 | 0.614 |
| 5 | 0.777 | 0.547 | 0.470 | 1.756 | 0.665 |

Cổng quên quanh 0.76–0.88 nghĩa là phần lớn cell state được giữ; `|c|` tăng đơn điệu từ 0.645 lên 1.756 cho thấy bộ nhớ tích lũy. Đây là hành vi mong muốn, và cũng cảnh báo: cell state có thể tăng không giới hạn nếu cổng quên luôn cao và đầu vào mạnh.

Pooling trên cùng chuỗi hidden state:

| Pooling | Vector |
| --- | --- |
| max | `[-0.024, -0.157, 0.085, -0.182]` |
| mean | `[-0.246, -0.253, -0.195, -0.269]` |
| last | `[-0.267, -0.371, -0.356, -0.326]` |

Ba vector khác nhau. Max giữ tín hiệu mạnh nhất ở bất kỳ vị trí nào; mean làm mượt; last chỉ nhìn cuối câu. Chọn pooling là chọn giả định về việc thông tin nằm ở đâu — không có lựa chọn "đúng" tuyệt đối.

So sánh suy giảm gradient:

| Bước | RNN 0.90^n | LSTM f=0.95^n | LSTM f=0.99^n |
| --- | --- | --- | --- |
| 10 | 3.49e−01 | 5.99e−01 | 9.04e−01 |
| 50 | 5.15e−03 | 7.69e−02 | 6.05e−01 |
| 100 | 2.66e−05 | 5.92e−03 | 3.66e−01 |
| 200 | 7.06e−10 | 3.51e−05 | 1.34e−01 |

LSTM không "nhớ vô hạn"; nó thay phép nhân ma trận lặp bằng cổng quên gần 1. Với f=0.99, tín hiệu còn 0.366 sau 100 bước; với f=0.95 chỉ còn 0.0059. Toàn bộ khác biệt giữa "nhớ được" và "quên sạch" nằm ở con số này.

Chi phí song song:

| Độ dài | CNN: lượt tính | RNN: bước nối tiếp |
| --- | --- | --- |
| 100 | 1 | 100 |
| 1000 | 1 | 1000 |
| 10000 | 1 | 10000 |

Đây là lý do kiến trúc attention thay được recurrence: không chỉ chính xác hơn mà còn song song hóa được theo độ dài.

### 4.5 Kết luận bài

TextCNN và RNN là hai cách bổ sung thứ tự vào biểu diễn: một bằng cửa sổ cục bộ học được, một bằng trạng thái tuần tự có bộ nhớ. Cả hai đều có trần: CNN không thấy xa, RNN không song song và vẫn khó truyền phụ thuộc rất xa. Attention giải cả hai, nhưng các kiến trúc này vẫn là lựa chọn đúng khi ràng buộc là phần cứng, streaming hoặc dữ liệu ít.

## Mạch kiến thức của bốn bài

1. **Bài 05** biến văn bản thành quyết định phân loại: biểu diễn + mô hình tuyến tính + metric đúng.
2. **Bài 06** nâng từ nhãn toàn câu lên nhãn từng token, và từ "đúng câu" lên "đúng span".
3. **Bài 07** thêm cấu trúc ngữ pháp: token nhận tag, câu nhận cây; trích xuất trở thành bài toán trên cạnh của cây.
4. **Bài 08** thay đặc trưng thủ công bằng kiến trúc học biểu diễn có thứ tự: cửa sổ (CNN) hoặc trạng thái (RNN).

Chuỗi này cho thấy một trục tiến hóa rõ: **từ rời rạc → có cấu trúc → có thứ tự → có ngữ cảnh**. Mỗi bước không xóa bước trước. TF-IDF vẫn thắng khi tín hiệu là từ khóa; gazetteer vẫn thắng khi miền hẹp và cần latency thấp; POS/dependency vẫn là hạ tầng của trích xuất; CNN/RNN vẫn đúng trên edge. Câu hỏi kỹ thuật không phải "mô hình nào mới nhất" mà "ràng buộc nào đang chi phối".

Một mẫu chung xuất hiện ở cả bốn bài: **con số tổng che lỗi cụ thể**. Accuracy 0.9 của sentiment che F1 lớp thiểu số bằng 0; token accuracy 0.875 của NER che PERSON F1 bằng 0; HMM 1.0 trên test nhỏ không ngăn nó sai `chased → AUX` ở câu OOV; gradient 0.9^100 "gần 0" chỉ có ý nghĩa khi so với 0.99^100. Đọc output theo từng nhóm, từng lỗi, từng bước — đó là kỹ năng chính của tuần này.

## Bảng tổng hợp kết quả thực hành

| Bài | Phần tự viết | Output tiêu biểu |
| --- | --- | --- |
| 05 | Tokenize + negate + bigram, NB, 4 biến thể, probe có kiểm soát, logistic regression, lệch lớp | Unigram 0.800; probe 0.500 → 1.000; `NOT_bad` +0.282, `NOT_worth` −0.277; accuracy 0.900 / macro-F1 0.474 |
| 06 | BIO/BILOU, gazetteer + luật, entity-level F1, Viterbi có ràng buộc, nested entity | Token 0.875 vs entity F1 0.750; argmax `O O I-GPE I-GPE O` → constrained `O B-GPE I-GPE I-GPE O`; multi-pass giữ cả ORG và FAC |
| 07 | MFT, HMM + Viterbi, per-tag F1, transition, dependency tree, SVO, passive, PP-attachment | MFT 0.938 vs HMM 1.000; transition `NOUN → VERB 0.41`; SVO `(company, bought, startup)`; passive trả `(None, bought, None)` |
| 08 | Conv1D + ReLU + max-pool, detector thủ công, bất biến vị trí, LSTM gates, pooling, gradient | Pooled `[1.0, 0.0, 1.0]`; max giữ nguyên khi đảo vị trí; RNN 2.66e−5 vs LSTM 0.366 sau 100 bước |

## Trạng thái chạy

- Cả bốn notebook chạy end-to-end bằng `nbconvert --execute`, mỗi notebook 8 cell (3 code cell), `execution_count` 1–3.
- Cả bốn `code/main.py` gốc chạy trong notebook với `Exit code: 0`.
- Không có cell lỗi ở lần chạy cuối; output đã lưu trong notebook.
- Không cần API key, không tải model ngoài; chỉ dùng stdlib + NumPy (đúng allowlist của repo).
- Các phần "Use It" trong tài liệu gốc (scikit-learn, spaCy, sklearn-crfsuite, Hugging Face) **không** chạy trong notebook để giữ khả năng chạy offline; notebook thay bằng cài đặt cơ chế tương đương.

## Việc nên làm tiếp

1. Chạy lại bài 05 trên một split công khai nhỏ (IMDb/SST-2) để thấy bigram + negation thực sự tạo chênh lệch bao nhiêu khi có đủ dữ liệu.
2. Với bài 06, cài `seqeval` và so entity F1 của notebook với một mô hình spaCy trên cùng 50 câu để kiểm tra trực giác "gazetteer yếu ở domain shift".
3. Với bài 07, dùng spaCy `en_core_web_sm` trên 20 câu để đối chiếu dependency tree dựng tay, đặc biệt là passive và PP-attachment.
4. Với bài 08, thay TextCNN NumPy bằng PyTorch `nn.Conv1d`/`nn.LSTM`, huấn luyện trên cùng một tập nhỏ và so max/mean/last pooling bằng số liệu thật.
5. Ghi lại các con số vào `PROJECTS-LEARNING.md` nếu các bài này được dùng làm nền cho project tuần sau.

## Ghi chú về giới hạn của báo cáo

Các corpus trong bốn notebook đều là dữ liệu đồ chơi do người viết tạo, kích thước 6–12 câu. Mọi con số accuracy/F1 chỉ minh họa cơ chế, không phải ước lượng chất lượng mô hình ngoài đời. Riêng bài 08 dùng trọng số đặt tay/cố định, không có huấn luyện. Khi cần so sánh với baseline trong tài liệu, phải chạy lại baseline trên đúng pipeline và đúng split, không dùng con số trong paper.
