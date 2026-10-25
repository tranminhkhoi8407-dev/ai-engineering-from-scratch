# Báo cáo học tập tuần 1

### Bài 1 — Dev Environment: Các lớp của môi trường phát triển

[Tài liệu bài học](phases/00-setup-and-tooling/01-dev-environment/docs/en.md)

Môi trường phát triển AI được tổ chức thành bốn lớp phụ thuộc nhau:

1. **Nền tảng hệ thống:** hệ điều hành, shell, Git, editor và driver thiết bị.
2. **Công cụ quản lý gói:** công cụ cài đặt thư viện và tạo môi trường như uv, pnpm, cargo hoặc công cụ tương ứng của Julia.
3. **Runtime ngôn ngữ:** môi trường thực thi Python, Node.js, Rust hoặc Julia.
4. **Thư viện AI/ML:** các thư viện được chương trình sử dụng ở tầng trên.

Cài đặt và kiểm tra từ dưới lên giúp xác định lỗi nằm ở tầng nào. Ví dụ, thư viện AI không thể sử dụng đúng nếu runtime hoặc driver nền tảng chưa phù hợp. Môi trường của khóa học có thể sử dụng nhiều ngôn ngữ, nhưng mỗi bài chỉ cần các công cụ được chỉ định cho lộ trình đang học; cài tất cả công cụ từ đầu không phải là yêu cầu.

Bài cũng phân biệt CPU với các lựa chọn tăng tốc phần cứng. GPU NVIDIA sử dụng CUDA; máy Apple Silicon có thể sử dụng MPS qua Metal. Nếu không có GPU, nhiều bài vẫn chạy được trên CPU; Colab hoặc máy GPU đám mây là phương án khi bài toán lớn hơn. Script kiểm tra theo route là một dạng preflight: nó xác nhận các điều kiện bắt buộc của lộ trình và chỉ ra bước tiếp theo, thay vì coi mọi công cụ dùng ở giai đoạn sau là điều kiện để bắt đầu.

**Ý nghĩa lý thuyết:** Môi trường là một phần của điều kiện thí nghiệm. Ghi nhận runtime, thư viện và thiết bị giúp giải thích vì sao cùng một code có thể chạy khác nhau trên hai máy.

### Bài 2 — Git & Collaboration: Mô hình quản lý phiên bản

[Tài liệu bài học](phases/00-setup-and-tooling/02-git-and-collaboration/docs/en.md)

Git theo dõi lịch sử thay đổi thông qua các snapshot được gọi là commit. Một thay đổi đi qua ba khu vực chính: thư mục làm việc, staging area và kho Git cục bộ. Lệnh add chọn phần thay đổi sẽ đưa vào snapshot; commit lưu snapshot cùng thông tin mô tả; push chuyển commit lên remote để sao lưu hoặc chia sẻ. Fetch và pull dùng để nhận thay đổi từ remote.

Branch là một con trỏ có thể di chuyển theo các commit mới. Tạo nhánh cho phép thử nghiệm biệt lập mà không làm ảnh hưởng nhánh chính; merge đưa các thay đổi phù hợp từ nhánh này vào nhánh khác. Fork là bản sao repo trên tài khoản của người học, hữu ích khi họ không có quyền ghi vào repo gốc. File .gitignore loại khỏi lịch sử những nội dung không nên quản lý như checkpoint, dữ liệu lớn hoặc môi trường ảo.

**Ý nghĩa lý thuyết:** Quản lý phiên bản không chỉ là lưu file. Nó tạo khả năng xem lại quá trình thay đổi, so sánh các phương án, khôi phục trạng thái trước và phối hợp mà không phải ghi đè công việc của nhau.

### Bài 3 — GPU Setup & Cloud: Lựa chọn tài nguyên tính toán

[Tài liệu bài học](phases/00-setup-and-tooling/03-gpu-setup-and-cloud/docs/en.md)

CPU và GPU phù hợp với các kiểu công việc khác nhau. CPU xử lý linh hoạt nhiều loại tác vụ; GPU có nhiều đơn vị tính toán song song nên thường phù hợp với phép toán ma trận và tensor lớn, vốn xuất hiện nhiều trong học sâu. Với bài toán nhỏ, thời gian truyền dữ liệu hoặc khởi chạy phép tính trên GPU có thể làm giảm lợi ích tăng tốc, vì vậy cần đo trên công việc thực tế thay vì mặc định rằng GPU luôn nhanh hơn.

Bài học phân loại ba lựa chọn sử dụng GPU: thiết bị NVIDIA cục bộ, Google Colab và máy GPU thuê trên cloud. Cách kiểm tra gồm xem thông tin thiết bị bằng nvidia-smi và kiểm tra khả năng CUDA qua PyTorch. Khi đo thời gian GPU, cần đồng bộ thiết bị quanh phép đo vì các phép tính GPU có thể được xếp hàng thực thi không đồng bộ.

VRAM giới hạn kích thước dữ liệu và mô hình có thể đặt trên GPU. Quy tắc sơ bộ của bài học là trọng số fp16 cần khoảng 2 byte cho mỗi tham số. Đây chỉ là ước lượng phần trọng số: bộ nhớ cho activation, gradient, optimizer và dữ liệu đầu vào cũng cần được tính đến. Vì vậy, giảm batch size, dùng độ chính xác thấp hơn hoặc chuyển sang máy có VRAM lớn hơn có thể là các lựa chọn khi gặp giới hạn bộ nhớ.

**Ý nghĩa lý thuyết:** Chọn tài nguyên là bài toán cân bằng giữa loại phép tính, bộ nhớ, thời gian và chi phí. CPU vẫn phù hợp để học và thử code; GPU hoặc cloud được dùng khi workload tạo ra nhu cầu rõ ràng.

### Bài 4 — APIs & Keys: Mô hình request–response và bảo vệ thông tin xác thực

[Tài liệu bài học](phases/00-setup-and-tooling/04-apis-and-keys/docs/en.md)

Một lời gọi API thường gồm bốn thành phần: địa chỉ endpoint, thông tin xác thực, nội dung request và nội dung response. Chương trình gửi request qua HTTP đến máy chủ; máy chủ xử lý yêu cầu rồi trả về response, thường có cấu trúc JSON. Trong dịch vụ mô hình ngôn ngữ, request thường chỉ định model, giới hạn đầu ra và các message cần gửi.

API key xác định tài khoản và cho phép chương trình sử dụng dịch vụ. Vì vậy, key nên được lấy từ biến môi trường hoặc file .env đã được loại khỏi Git, thay vì ghi thẳng vào mã nguồn. SDK cung cấp giao diện thuận tiện và che bớt chi tiết HTTP; gọi HTTP trực tiếp cho thấy request và response ở dạng nền tảng, giúp lần theo nguyên nhân khi tích hợp gặp lỗi. Các nhà cung cấp có thể dùng endpoint, tên model và cấu trúc dữ liệu khác nhau dù cùng tuân theo mẫu request–response.

Bài giới thiệu thêm lỗi xác thực, giới hạn tốc độ và khái niệm token trong ngữ cảnh API. Token ở đây là đơn vị tính lượng nội dung được gửi và nhận, thường có ảnh hưởng đến hạn mức hoặc chi phí. Streaming là cách nhận kết quả thành từng phần thay vì chờ toàn bộ response.

**Ý nghĩa lý thuyết:** Tích hợp API cần đồng thời quan tâm đến giao thức, cấu trúc dữ liệu, quyền truy cập và cách giữ bí mật. Hiểu HTTP bên dưới SDK giúp việc chẩn đoán có cơ sở hơn.

### Bài 5 — Jupyter Notebooks: Kernel, cell và trạng thái thực thi

[Tài liệu bài học](phases/00-setup-and-tooling/05-jupyter-notebooks/docs/en.md)

Notebook gồm các cell code và cell văn bản. Một kernel là tiến trình Python chạy ở phía sau, thực thi các cell và giữ biến trong bộ nhớ. Các cell dùng chung kernel nhưng có thể được chạy theo thứ tự tùy ý. Đặc điểm này giúp thử nghiệm từng bước nhanh, đồng thời tạo ra nguy cơ notebook phụ thuộc vào trạng thái cũ: một biến có thể vẫn tồn tại dù cell tạo ra nó đã bị sửa hoặc xóa.

Magic command là lệnh riêng của Jupyter, dùng để điều khiển môi trường notebook, chẳng hạn đo thời gian hoặc hiển thị biểu đồ. JupyterLab, Jupyter Notebook, VS Code và Colab là các giao diện khác nhau có thể làm việc với định dạng notebook. Colab giảm yêu cầu cài đặt tại máy nhưng phiên làm việc có giới hạn và file không tự động tồn tại sau mỗi phiên nếu không lưu ra nơi khác.

Notebook phù hợp cho khám phá dữ liệu, trực quan hóa, giải thích và prototype. Script phù hợp hơn với tác vụ cần chạy lặp lại, dùng trong pipeline hoặc đóng gói thành thư viện. Khởi động lại kernel rồi chạy toàn bộ cell theo thứ tự là cách kiểm tra xem notebook có thật sự tự chứa và tái chạy được hay không.

**Ý nghĩa lý thuyết:** Notebook lưu cả code lẫn kết quả, nhưng thứ tự lưu cell không đồng nghĩa với thứ tự thực thi. Muốn chia sẻ kết quả đáng tin cậy cần quản lý trạng thái kernel và kiểm tra lại toàn bộ quy trình.

### Bài 6 — Python Environments: Cô lập dependency và tái lập cài đặt

[Tài liệu bài học](phases/00-setup-and-tooling/06-python-environments/docs/en.md)

Dependency là thư viện mà dự án cần để chạy; dependency bắc cầu là thư viện được một dependency khác yêu cầu. Các dự án AI có thể cần phiên bản framework hoặc CUDA khác nhau. Nếu tất cả package được cài vào cùng Python toàn hệ thống, nâng cấp cho dự án này có thể làm dự án khác hỏng. Virtual environment giải quyết vấn đề bằng cách cô lập interpreter và package theo dự án.

uv và venv có thể tạo môi trường Python riêng; conda có thể quản lý thêm một số dependency ngoài Python như thư viện hệ thống và CUDA toolkit. Bài khuyến nghị tránh trộn công cụ quản lý package một cách tùy tiện trong cùng môi trường, đặc biệt là pip và conda, vì chúng có thể không đồng bộ cách giải quyết dependency.

pyproject.toml mô tả metadata và dependency của dự án, có thể chia các nhóm tùy chọn như gói cơ bản, gói GPU hoặc SDK. Lockfile ghi lại phiên bản cụ thể của cả dependency trực tiếp và bắc cầu, nhờ đó các máy khác có thể cài một tập package tương ứng. Môi trường ảo không nên được commit vì nó chứa nhiều file phụ thuộc máy; thay vào đó cần lưu khai báo dependency và lockfile. Với GPU, phiên bản CUDA mà framework được dựng cùng phải tương thích với driver của hệ thống.

**Ý nghĩa lý thuyết:** Tái lập môi trường đòi hỏi cả sự cô lập lẫn mô tả phiên bản. Chỉ ghi tên thư viện mà không ghi phiên bản có thể chưa đủ để tái tạo một thí nghiệm.

### Bài 7 — Docker for AI: Image, container, volume và dịch vụ

[Tài liệu bài học](phases/00-setup-and-tooling/07-docker-for-ai/docs/en.md)

Docker đóng gói ứng dụng, runtime, thư viện và công cụ hệ thống. Image là mẫu bất biến dùng để tạo môi trường; container là một tiến trình đang chạy từ image; Dockerfile mô tả các lớp tạo image. Container chia sẻ kernel của hệ điều hành host, vì vậy nhẹ hơn máy ảo đầy đủ nhưng vẫn cung cấp filesystem và mạng được cô lập.

Dockerfile được tạo theo từng layer. Khi một layer không thay đổi, Docker có thể dùng lại cache để giảm thời gian build. Chọn base image là quyết định về những thành phần có sẵn: image development có compiler và toolkit để dựng thư viện; image runtime nhẹ hơn để chạy phần mềm đã được xây dựng; image CPU-only có thể nhỏ hơn nữa. Trong môi trường GPU, container cần runtime/toolkit thích hợp để sử dụng thiết bị NVIDIA trên host; container không thay thế driver vật lý.

Container có vòng đời riêng, nên volume được dùng để giữ dữ liệu model, dataset hoặc kết quả bên ngoài filesystem tạm của container. Docker Compose mô tả và khởi chạy nhiều dịch vụ cùng lúc, ví dụ môi trường phát triển AI và cơ sở dữ liệu vector. Các dịch vụ trong cùng mạng Compose có thể giao tiếp bằng tên dịch vụ.

**Ý nghĩa lý thuyết:** Docker giúp chuẩn hóa runtime và dependency giữa các máy; volume tách dữ liệu lâu dài khỏi vòng đời container. Image lớn và cấu hình GPU vẫn cần được lựa chọn phù hợp với nền tảng thực tế.

### Bài 8 — Editor Setup: Ngôn ngữ máy chủ và công cụ hỗ trợ lập trình

[Tài liệu bài học](phases/00-setup-and-tooling/08-editor-setup/docs/en.md)

Editor hiện đại không chỉ là nơi soạn thảo văn bản. Các extension kết nối editor với ngôn ngữ, notebook, debugger, hệ thống quản lý phiên bản và máy từ xa. Language Server Protocol (LSP) là chuẩn giao tiếp để editor nhận thông tin kiểu dữ liệu, gợi ý hoàn thành và chẩn đoán từ một language server. Pylance cung cấp các chức năng kiểu này cho Python.

Các công cụ trong bài đảm nhiệm vai trò khác nhau: type checking tìm một số lỗi kiểu trước khi chạy; linter báo vấn đề về code; formatter chuẩn hóa cách trình bày; debugger cho phép dừng và quan sát trạng thái chương trình; extension Jupyter hỗ trợ notebook; GitLens giúp xem lịch sử thay đổi. Format-on-save và autosave giảm thao tác lặp lại, còn notebook output scrolling giúp quản lý output dài.

Remote SSH kết nối editor cục bộ với filesystem, terminal và tiến trình trên máy Linux/GPU từ xa. Nhờ đó, người học có thể chỉnh sửa và gỡ lỗi code ở máy đang tính toán mà không phải sao chép toàn bộ dự án qua lại. VS Code là một lựa chọn được bài khuyến nghị; các editor dựa trên VS Code có thể dùng chung nhiều cấu hình, còn người đã quen Vim/Neovim có thể giữ công cụ đó.

**Ý nghĩa lý thuyết:** Editor là một lớp tích hợp các công cụ phát triển. Cấu hình tốt giúp phát hiện lỗi sớm và giảm ma sát, nhưng không thay thế việc hiểu code hoặc kiểm tra hành vi thật của chương trình.

### Bài 9 — Data Management: Vòng đời, định dạng và chia tập dữ liệu

[Tài liệu bài học](phases/00-setup-and-tooling/09-data-management/docs/en.md)

Một pipeline dữ liệu thường đi qua các bước tải, lưu cache, đọc hoặc stream, chuyển đổi định dạng, chia tập và đưa dữ liệu vào huấn luyện. Cache lưu bản sao cục bộ để tránh tải lại nguồn đã dùng. Streaming đọc dữ liệu từng mẫu thay vì tải toàn bộ trước; cách này giảm nhu cầu lưu trữ và giữ lượng dữ liệu trong bộ nhớ ổn định hơn khi xử lý tập rất lớn.

Các định dạng có ưu điểm khác nhau. CSV dễ đọc và trao đổi nhưng thường lớn, đọc chậm; JSON linh hoạt với dữ liệu lồng nhau nhưng cũng không tối ưu cho phân tích khối lượng lớn; Parquet là định dạng cột, thường gọn và phù hợp truy vấn phân tích; Arrow hỗ trợ xử lý dữ liệu dạng cột trong bộ nhớ. Không có một định dạng duy nhất phù hợp cho mọi giai đoạn.

Train là tập dùng để học tham số mô hình; validation dùng để theo dõi và lựa chọn cấu hình trong quá trình phát triển; test dành cho đánh giá cuối. Seed cố định giúp phép chia có thể được lặp lại. Các phép biến đổi có học thống kê, chẳng hạn chuẩn hóa, cần được fit trên train rồi áp dụng cho validation/test. Nếu thông tin từ test lọt vào bước huấn luyện hoặc tiền xử lý, đánh giá có thể lạc quan hơn khả năng tổng quát hóa thật; đây là data leakage.

Git không thích hợp để lưu mọi dữ liệu và model lớn. .gitignore phù hợp với file có thể tải lại; Git LFS lưu nội dung lớn ngoài object thông thường của Git nhưng vẫn giữ pointer; DVC thêm quy trình quản lý phiên bản dữ liệu và liên kết tới storage từ xa.

**Ý nghĩa lý thuyết:** Kết quả mô hình chỉ có ý nghĩa nếu dữ liệu được quản lý và đánh giá đúng. Reproducibility bao gồm cả nguồn dữ liệu, phiên bản, phép chia và các bước tiền xử lý.

### Bài 10 — Terminal & Shell: Luồng lệnh và tiến trình dài hạn

[Tài liệu bài học](phases/00-setup-and-tooling/10-terminal-and-shell/docs/en.md)

Shell diễn giải lệnh của người dùng và kết nối các chương trình nhỏ thành quy trình. Pipe chuyển đầu ra của chương trình này thành đầu vào của chương trình khác; redirect ghi stdout hoặc stderr ra file; nhờ đó log có thể được lọc, lưu và phân tích mà không cần mở toàn bộ nội dung bằng giao diện đồ họa.

Tiến trình chạy nền cho phép terminal tiếp tục nhận lệnh. Tuy nhiên, chạy bằng dấu & không đảm bảo tác vụ còn sống sau khi terminal đóng. nohup giúp tác vụ không bị kết thúc khi mất terminal nhưng không cung cấp cách quay lại phiên tương tác. tmux tạo phiên terminal có thể detach và attach lại, đồng thời chia pane để chạy nhiều công việc như huấn luyện, theo dõi GPU và đọc log.

Bài cũng trình bày quan sát tài nguyên bằng htop, nvtop và nvidia-smi; kết nối máy từ xa bằng SSH; sao chép file bằng scp; đồng bộ thư mục bằng rsync. Alias lưu các lệnh thường dùng dưới tên ngắn để giảm gõ lặp lại, nhưng cần hiểu lệnh gốc trước khi dùng alias.

**Ý nghĩa lý thuyết:** Shell hỗ trợ điều phối quy trình tính toán, còn tmux và log giúp tách vòng đời job khỏi vòng đời cửa sổ terminal. Điều này quan trọng khi huấn luyện kéo dài hoặc chạy trên cloud.

### Bài 11 — Linux for AI: Cấu trúc và vận hành máy chủ

[Tài liệu bài học](phases/00-setup-and-tooling/11-linux-for-ai/docs/en.md)

Linux tổ chức file dưới một thư mục gốc duy nhất là /. Thư mục home thường chứa dự án và dữ liệu của người dùng; tmp chứa dữ liệu tạm; etc chứa cấu hình; var/log chứa log; proc và sys cung cấp thông tin ảo về tiến trình, kernel và phần cứng. Hiểu cấu trúc này giúp tìm file cấu hình, log và dữ liệu trên máy chủ không có giao diện đồ họa.

Quyền file được chia theo chủ sở hữu, nhóm và những người còn lại, với các quyền đọc, ghi và thực thi. chmod thay đổi quyền; chown thay đổi chủ sở hữu; sudo chạy một lệnh với quyền quản trị. Nên dùng quyền quản trị chỉ khi cần. Lệnh xóa đệ quy không có thùng rác, vì vậy phải kiểm tra đường dẫn trước khi thực hiện.

apt cài package hệ thống; ps, htop và kill hỗ trợ quan sát hoặc dừng tiến trình; systemd quản lý dịch vụ chạy nền; df và du cho biết dung lượng đĩa và thư mục; curl/wget tải nội dung; scp/rsync truyền dữ liệu. Trên Windows, WSL2 cung cấp môi trường Linux có kernel riêng; khi chuyển giữa Windows, macOS và Linux cần chú ý đường dẫn, shell mặc định, tên package, phân biệt chữ hoa/thường và line ending.

**Ý nghĩa lý thuyết:** Làm việc trên máy GPU từ xa đòi hỏi hiểu cả file system, quyền, tiến trình, dịch vụ và dung lượng. Những thao tác này giúp người học tránh để máy tính phí chạy khi job đã lỗi hoặc ổ đĩa đã đầy.

### Bài 12 — Debugging and Profiling: Tìm lỗi và nút thắt có bằng chứng

[Tài liệu bài học](phases/00-setup-and-tooling/12-debugging-and-profiling/docs/en.md)

Gỡ lỗi AI có thể xem ở ba tầng. Tầng Python gồm breakpoint, log, thời gian và bộ nhớ của chương trình. Tầng tensor gồm shape, dtype, device và giá trị NaN/Inf. Tầng huấn luyện gồm loss, gradient, activation và xu hướng tổng quát hóa. Lỗi ở tầng sau thường bắt nguồn từ một giả định sai ở tầng trước, vì vậy cần kiểm tra dữ liệu và phép biến đổi trước khi kết luận mô hình cần thay đổi.

Shape mismatch xảy ra khi tensor đầu vào không có dạng mà một phép toán hoặc layer mong đợi. Tensor sai device có thể gây lỗi hoặc làm tính toán không đúng quy trình mong muốn. NaN loss có thể đến từ learning rate quá cao, chia cho 0, log giá trị không hợp lệ hoặc gradient bùng nổ. Data leakage làm kết quả đánh giá có vẻ tốt hơn khả năng thực tế. Đây là những ví dụ về lỗi im lặng: chương trình có thể tiếp tục chạy mà vẫn tạo ra kết luận sai.

Breakpoint cho phép dừng có điều kiện để kiểm tra trạng thái; logging ghi sự kiện kèm thời điểm và mức độ; timer và cProfile tìm phần code chiếm thời gian; tracemalloc hoặc profiler bộ nhớ giúp tìm nơi cấp phát; công cụ PyTorch cho biết bộ nhớ GPU đã cấp phát hoặc dự trữ. TensorBoard trình bày loss, histogram trọng số và gradient để quan sát quá trình học. Profiling trả lời câu hỏi phần nào tốn tài nguyên; chỉ sau khi biết điểm nghẽn mới nên chọn cách tối ưu.

**Ý nghĩa lý thuyết:** Một đường loss đẹp chưa đủ chứng minh mô hình học đúng. Cần kết hợp kiểm tra đầu vào, trạng thái tensor, log và chỉ số huấn luyện để xác minh kết quả.

### Bài 13 — Python for AI Engineering: Các thành phần Python dùng trong pipeline

[Tài liệu bài học](phases/00-setup-and-tooling/13-python-for-ai-engineering/docs/en.md)

Bài học hệ thống hóa các cấu trúc Python thường xuất hiện trong code AI: collection, unpacking, comprehension, hàm, tham số, closure, class và dataclass, iterator, generator, context manager, exception, type hint, module, main guard, CSV, JSON và công cụ trong itertools. Một dataclass giúp biểu diễn bản ghi có cấu trúc; type hint mô tả kiểu dự kiến; context manager gom phần mở và đóng tài nguyên để thao tác an toàn hơn.

Python quản lý biến theo tham chiếu tới object. Vì vậy, hai biến có thể cùng trỏ tới một list; sửa list qua biến này làm thay đổi nội dung nhìn thấy qua biến kia. Mutable default argument cũng có thể được dùng lại giữa nhiều lần gọi hàm. Generator tạo dữ liệu theo yêu cầu và thường chỉ tiêu thụ được một lần, nên cần tạo lại cho mỗi epoch. Những trường hợp này có thể làm sai kết quả mà không gây exception.

Phần pipeline ghép các ý trên thành một luồng xử lý: đọc CSV/JSON vào kiểu Record; chia train/test bằng generator ngẫu nhiên riêng có seed; tạo mini-batch; tính dot product và thống kê; fit standardizer trên train rồi áp dụng cùng tham số sang dữ liệu khác; cuối cùng dùng centroid để phân loại. Thứ tự này vừa thể hiện cấu trúc code vừa nhắc lại nguyên tắc tránh leakage.

Vòng lặp Python thuần hữu ích để hiểu phép tính, nhưng mỗi phần tử có chi phí object và vòng lặp thông dịch. NumPy lưu dữ liệu số trong mảng kiểu cố định và thực hiện phép toán theo khối trong code đã biên dịch; broadcasting cho phép kết hợp các mảng có shape tương thích mà không phải viết vòng lặp tường minh. Khi giảm chiều để tính chuẩn hóa hoặc chuẩn vector, giữ đúng shape là cần thiết để phép toán áp dụng theo trục mong muốn.

**Ý nghĩa lý thuyết:** Hiểu idiom Python giúp người học đọc được code mô hình mà không bị vướng cú pháp; hiểu chi phí thực thi giúp biết khi nào giữ code rõ ràng bằng Python thuần và khi nào chuyển phần tính toán số sang thư viện mảng.

## 3. Mối liên hệ giữa các bài

Mười ba bài tạo thành một chuỗi phụ thuộc thay vì các chủ đề rời rạc:

1. **Chuẩn bị máy và công cụ:** Dev Environment xác định các lớp phần mềm; Python Environments cô lập dependency; Editor Setup cung cấp giao diện viết, chạy và debug.
2. **Tổ chức công việc:** Git lưu lịch sử thay đổi; Terminal, Shell và Linux giúp vận hành tiến trình và dữ liệu trên máy cá nhân hoặc máy chủ.
3. **Chạy chương trình:** GPU Setup & Cloud giúp chọn nơi tính toán; Docker đóng gói runtime khi cần chia sẻ một cấu hình nhất quán.
4. **Kết nối và dữ liệu:** APIs & Keys quản lý giao tiếp với dịch vụ ngoài; Data Management quản lý dữ liệu đầu vào, cache, định dạng và phép chia đánh giá.
5. **Thử nghiệm và kiểm chứng:** Jupyter hỗ trợ khám phá; Python for AI Engineering tạo pipeline có cấu trúc; Debugging and Profiling xác minh code, tensor, hiệu năng và hành vi huấn luyện.

Ví dụ, một thí nghiệm huấn luyện có thể bắt đầu bằng việc kiểm tra môi trường và tạo nhánh Git; cài dependency trong môi trường riêng; đọc dữ liệu và chia train/validation/test có seed; khám phá trong notebook; chuyển code ổn định sang script; chạy trên GPU nếu cần; dùng tmux và log để theo dõi; dùng debugger hoặc profiler khi có dấu hiệu sai; sau cùng lưu thay đổi code và cấu hình cần thiết để có thể xem lại.

## 4. Nhận xét tổng hợp

- **Khả năng tái lập cần nhiều hơn một seed.** Cần biết cả phiên bản dependency, runtime, thiết bị, dữ liệu, phép chia, tiền xử lý và lệnh chạy.
- **Tính đúng đắn phụ thuộc vào cách tổ chức dữ liệu.** Nếu test bị dùng trong quá trình huấn luyện hoặc chuẩn hóa, chỉ số đánh giá không còn phản ánh độc lập khả năng tổng quát hóa.
- **An toàn thông tin thuộc về thiết kế workflow.** API key phải được tách khỏi code; file lớn, model và dữ liệu cần có nơi lưu phù hợp thay vì đưa tất cả vào Git.
- **Tài nguyên phải được đo và quản lý.** GPU, VRAM, đĩa, bộ nhớ và thời gian chạy đều có giới hạn; log và profiler cung cấp bằng chứng để xử lý các giới hạn đó.
- **Công cụ không thay cho tư duy.** Notebook, SDK, Docker, editor và debugger giúp thực hiện công việc, nhưng người học vẫn cần hiểu trạng thái chương trình, giao thức, dữ liệu và giả định của từng phép tính.

## 5. Kết luận

Phase 00 xây dựng nền móng vận hành cho các nội dung AI phía sau. Người học đi từ việc hiểu cấu trúc môi trường, version control và tài nguyên phần cứng đến việc quản lý API, notebook, dependency, container, dữ liệu và máy Linux. Hai bài cuối cùng đưa quy trình về phía code: một bài cung cấp phương pháp kiểm tra lỗi và hiệu năng; bài còn lại hệ thống hóa Python để xây dựng pipeline dữ liệu và phép tính cơ bản.

Thông điệp tổng quát của tuần học là một thí nghiệm AI đáng tin cậy cần được chuẩn bị, cô lập, ghi nhận và kiểm chứng. Khi nền tảng này vững, thời gian ở các phase tiếp theo có thể tập trung vào toán học, thuật toán và kiến trúc mô hình thay vì xử lý những lỗi workflow có thể phòng tránh.

## Tài liệu tham khảo trong repo

- [Phase 00 — README](phases/00-setup-and-tooling/README.md)
- [ROADMAP — thời lượng và trạng thái các bài](ROADMAP.md)
- Các tài liệu bài học được liên kết ngay dưới tiêu đề từng bài.
