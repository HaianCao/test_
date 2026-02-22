<div align="center">
  <img src="./assets/banner1.jpg" width="100%">
</div>

# 🚀 DỰ ĐOÁN HÀNH VI NGƯỜI DÙNG (USER BEHAVIOR PREDICTION)

Một doanh nghiệp lớn đang đối mặt với bài toán tối ưu hóa chi phí vận hành kho bãi. Để giải quyết vấn đề này, doanh nghiệp mong muốn dự đoán trước các hành vi mua sắm của khách hàng dựa trên chuỗi hành động tương tác của họ trong quá khứ, từ đó giúp đội ngũ Sales phân tích và có chiến lược nhập hàng, quản lý tồn kho hiệu quả.

Thí sinh, với vai trò một Data Scientist, cần giải quyết bài toán **Dự đoán hành vi người dùng**.

---

# Nội dung
- [Mô tả bài toán](#mô-tả-bài-toán)
    - [1. Đặc điểm bài toán](#1-đặc-điểm-bài-toán)
    - [2. Cấu trúc dữ liệu](#2-cấu-trúc-dữ-liệu)
    - [3. Kết quả dự đoán](#3-kết-quả-dự-đoán)
    - [4. Chỉ số đánh giá](#3-chỉ-số-đánh-giá)
- [Starter Kit](#starter-kit)
- [Các mốc thời gian quan trọng](#các-mốc-thời-gian-quan-trọng)
- [Cách thức tham gia](#cách-thức-tham-gia)
- [Quy định cuộc thi](#quy-định-cuộc-thi)
- [Tài nguyên](#tài-nguyên)
- [Ban tổ chức](#ban-tổ-chức)

# Mô tả bài toán

**Nhiệm vụ:**
Phân tích dữ liệu và xây dựng mô hình phân loại đa đầu ra (Multi-output Classifier) để dự đoán 6 thuộc tính hành vi độc lập của một khách hàng trong tương lai, dựa trên chuỗi hành động trong quá khứ của họ.

### 1. Đặc điểm bài toán

- **Dữ liệu đầu vào (Input)** là một chuỗi các hành động liên tiếp của khách hàng theo thời gian. Trong vòng 4 tuần, khách hàng có thể cập nhật giao dịch và thay đổi nội dung đơn hàng. Toàn bộ các hành động này đã được mã hóa thành các con số định danh nhằm bảo mật thông tin.
- **Dữ liệu đầu ra (Output)** gồm 6 biến mục tiêu độc lập nhằm hỗ trợ doanh nghiệp xác định xem mỗi ngày cần nhập và phân bổ những loại hàng hóa nào, từ đó tối ưu chi phí lưu kho và vận hành. Sáu biến này đại diện cho 6 khía cạnh hành vi khác nhau của người dùng và không có tính liên quan trực tiếp đến nhau. 

### 2. Cấu trúc dữ liệu

Dữ liệu đã được Ban tổ chức chia sẵn thành 3 tập độc lập: Tập huấn luyện (Train), Tập kiểm định (Validation), và Tập kiểm thử (Test), tương ứng với các tập dữ liệu **X_train.csv và Y_train.csv**, **X_val.csv và Y_val.csv**, **X_test.csv**. 

Mỗi dòng trong **tập X (Đầu vào)** sẽ tương ứng trực tiếp với một dòng ở **tập Y (Đầu ra)**, bao gồm 6 thuộc tính cần dự đoán. Mỗi dòng đại diện cho một phiên giao dịch khách hàng duy nhất.

- **Tập X (Đầu vào):** Chứa chuỗi các hành động của người dùng đã được mã hóa. Lưu ý: Độ dài chuỗi đầu vào của mỗi khách hàng là khác nhau (Variable-length sequence) do tần suất và số lượng tương tác của mỗi người dùng không đồng nhất.
- **Tập Y (Đầu ra):** Chứa 6 cột dữ liệu, tương ứng với 6 thuộc tính hành vi cần dự đoán. Dữ liệu đầu ra yêu cầu định dạng số nguyên không dấu 16-bit (UINT16). Nếu mô hình dự đoán ra số thực (float), thí sinh cần ép kiểu kết quả về UINT16.

> **LƯU Ý:**
> Để đảm bảo tính công bằng trong học thuật và tối ưu hiệu năng trên bảng xếp hạng, Ban tổ chức quy định rõ cách thức sử dụng các tập dữ liệu như sau:
> 1. **Giai đoạn Nghiên cứu, EDA và Đánh giá nội bộ (Bắt buộc cô lập):** Trong quá trình Phân tích Dữ liệu Khám phá (EDA), thiết lập quy trình Tiền xử lý (Feature Engineering) và tính toán các chỉ số (metrics) để báo cáo, thí sinh tuyệt đối không được gộp tập Train và Validation.
> 2. **Giai đoạn Tối ưu hóa điểm số trên Kaggle (Được phép gộp):** Sau khi đã chốt được kiến trúc mô hình và các tham số tối ưu (Hyperparameters) thông qua quá trình đánh giá ở Giai đoạn 1, để tạo ra file kết quả (Submission) nộp lên hệ thống Kaggle, thí sinh **ĐƯỢC PHÉP gộp chung** tập Train và Validation lại thành một tập dữ liệu lớn hơn. Các bạn có thể chạy lại toàn bộ quy trình tiền xử lý và huấn luyện (Retrain) trên tập dữ liệu tổng hợp này để mô hình học được nhiều thông tin nhất có thể trước khi dự đoán trên tập Test cuối cùng.

### 3. Kết quả dự đoán

Với mỗi `ID` trong tập X_test.csv, thí sinh cần dự báo 6 thuộc tính tương ứng. File nộp cần có định dạng `.csv` với tiêu đề cột chính xác như sau:
```csv
id,attr_1,attr_2,attr_3,attr_4,attr_5,attr_6
gpbfd,0,0,0,0,0,0
w22ee,0,0,0,0,0,0
wyw95,0,0,0,0,0,0
izx4w,0,0,0,0,0,0
c6o2d,0,0,0,0,0,0
```

### 4. Chỉ số đánh giá

Cuộc thi sử dụng chỉ số **Exact-Match Accuracy (Độ chính xác khớp tuyệt đối)** trên tập Test để đánh giá xếp hạng.

**Quy tắc tính điểm:**
Bài toán yêu cầu dự đoán một vector đầu ra gồm 6 chiều (tương ứng với 6 thuộc tính hành vi) cho mỗi người dùng.

**Một dự đoán chỉ được tính là Chính xác khi và chỉ khi mô hình dự đoán đúng đồng thời cả 6 giá trị của vector đầu ra**. Nếu mô hình dự đoán sai dù chỉ một trong 6 chiều, toàn bộ kết quả dự đoán của dòng dữ liệu đó sẽ bị đánh giá là Không chính xác.

**Công thức chỉ số Exact-Match Accuracy**
$$\text{Accuracy} = \frac{N_{acc}}{N}$$

**Trong đó:**
- **N_acc**: Số lượng mẫu dự đoán đúng hoàn toàn (khớp đồng thời cả 6 thuộc tính).
- **N**: Tổng số lượng mẫu trên tập Test.

# Starter Kit

Chúng tôi cung cấp **starter kit** cho bài toán để giúp người tham gia bắt đầu dễ dàng hơn:
- **Starter Kit**: [`starter_kit/`](./starter_kit/)

Và các công cụ hỗ trợ:
- **Script đánh giá**: [`evaluation_script/`](./evaluation_script/)
- **File nộp mẫu**: [`sample_submission_file/`](./sample_submission_file/)
- **Dữ liệu cuộc thi**: [`dataset/`](./dataset/) *(Lưu ý: Thư mục này được thiết lập bỏ qua trong `.gitignore`. Thí sinh cần tự tải dữ liệu từ nền tảng cuộc thi trên Kaggle và đặt vào thư mục này)*

# Các mốc thời gian quan trọng

# Cách thức tham gia

# Quy định cuộc thi

# Tài nguyên

- [Script đánh giá](./evaluation_script/README.md)
- [Thư mục dữ liệu](./dataset/README.md)
- [Thư mục nộp mẫu (Sample Submission)](./sample_submission_file/)
- [Starter Kit](./starter_kit/)
