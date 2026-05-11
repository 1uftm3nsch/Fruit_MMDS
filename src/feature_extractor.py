import cv2
import numpy as np


# Kích thước chuẩn cho mọi ảnh trước khi trích xuất đặc trưng.
# Dataset Fruits-360 có ảnh 100x100, nên query image cũng được đưa về cùng scale.
IMAGE_SIZE = (100, 100)

# Số bins của histogram texture LBP.
# LBP tạo mã từ 0 đến 255, nên dùng 256 bins để đếm đầy đủ các mẫu texture.
LBP_BINS = 256

# Tổng số chiều vector đặc trưng:
# 512 màu sắc (HSV 8x8x8) + 7 hình dáng (Hu Moments) + 256 texture (LBP).
FEATURE_DIMENSION = 512 + 7 + LBP_BINS


def normalize_histogram(hist):
    # Chuyển histogram về float32 và làm phẳng thành vector 1 chiều.
    hist = hist.astype(np.float32).flatten()

    # Tổng số lượng giá trị trong histogram.
    # Dùng để chuẩn hóa histogram về dạng tỉ lệ thay vì số đếm tuyệt đối.
    total = hist.sum()

    # Chỉ chia khi tổng > 0 để tránh lỗi chia cho 0.
    # Sau chuẩn hóa, histogram ít phụ thuộc vào kích thước/vùng ảnh được lấy.
    if total > 0:
        hist = hist / total
    return hist


def create_fruit_mask(image):
    """
    Tạo mask vùng quả để giảm ảnh hưởng của nền.
    Nếu tách nền không tốt thì dùng toàn bộ ảnh làm fallback.
    """
    # Chuyển sang grayscale để dùng threshold tách vùng sáng/tối.
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Làm mờ nhẹ để giảm nhiễu trước khi threshold.
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Chuyển sang HSV để lấy kênh saturation, giúp phát hiện vùng quả có màu nổi bật.
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Otsu tự tìm ngưỡng để tách quả khỏi nền, không cần cố định threshold = 240.
    _, otsu_mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Saturation mask giữ lại các vùng có màu rõ, hữu ích với ảnh nền không trắng.
    saturation_mask = cv2.inRange(hsv[:, :, 1], 25, 255)

    # Gộp hai mask để tận dụng cả thông tin sáng/tối và độ bão hòa màu.
    combined_mask = cv2.bitwise_or(otsu_mask, saturation_mask)

    # Morphology giúp xóa nhiễu nhỏ và lấp các lỗ nhỏ trong vùng quả.
    kernel = np.ones((3, 3), np.uint8)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Tìm các contour trong mask; mỗi contour là một vùng vật thể liên thông.
    contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        # Nếu không tìm được contour, dùng toàn ảnh để extractor vẫn chạy được.
        return np.full(gray.shape, 255, dtype=np.uint8)

    # Chọn contour lớn nhất vì giả định ảnh query có một quả chính.
    largest_contour = max(contours, key=cv2.contourArea)
    image_area = image.shape[0] * image.shape[1]
    contour_area = cv2.contourArea(largest_contour)

    # Nếu mask quá nhỏ hoặc gần như phủ toàn ảnh thì coi là tách nền thất bại.
    if contour_area < image_area * 0.03 or contour_area > image_area * 0.95:
        return np.full(gray.shape, 255, dtype=np.uint8)

    # Vẽ contour lớn nhất thành mask nhị phân: vùng quả = 255, nền = 0.
    mask = np.zeros(gray.shape, dtype=np.uint8)
    cv2.drawContours(mask, [largest_contour], -1, 255, thickness=cv2.FILLED)
    return mask


def calculate_lbp(gray):
    """
    Tính Local Binary Pattern (LBP) để mô tả texture bề mặt.
    Mỗi pixel được so sánh với 8 pixel lân cận để tạo mã từ 0 đến 255.
    """
    # Bỏ viền ngoài để mỗi pixel trung tâm luôn có đủ 8 hàng xóm xung quanh.
    center = gray[1:-1, 1:-1]

    # Mỗi pixel trong lbp lưu một mã 8-bit biểu diễn mẫu texture cục bộ.
    lbp = np.zeros_like(center, dtype=np.uint8)

    # 8 hướng hàng xóm quanh pixel trung tâm, đi theo chiều kim đồng hồ.
    neighbors = [
        (-1, -1), (-1, 0), (-1, 1),
        (0, 1), (1, 1), (1, 0),
        (1, -1), (0, -1),
    ]

    # Nếu pixel hàng xóm >= pixel trung tâm thì bật bit tương ứng trong mã LBP.
    for bit, (dy, dx) in enumerate(neighbors):
        neighbor = gray[1 + dy:gray.shape[0] - 1 + dy, 1 + dx:gray.shape[1] - 1 + dx]
        lbp |= ((neighbor >= center).astype(np.uint8) << bit)

    # Ảnh LBP này sẽ được đổi thành histogram ở phần ĐẶC TRƯNG 3.
    return lbp


def extract_features(image_path):
    """
    Trích xuất kết hợp Màu sắc (Color Histogram), Hình dáng (Hu Moments)
    và Kết cấu (LBP Texture).
    Trả về một vector đặc trưng duy nhất.
    """
    # Đọc ảnh
    image = cv2.imread(str(image_path))
    if image is None:
        return None

    # Chuẩn hóa kích thước để ảnh dataset và ảnh query có cùng scale.
    image = cv2.resize(image, IMAGE_SIZE, interpolation=cv2.INTER_AREA)

    # Tạo mask vùng quả để các đặc trưng ít bị ảnh hưởng bởi nền.
    mask = create_fruit_mask(image)

    # ==========================================
    # ĐẶC TRƯNG 1: MÀU SẮC (Color Histogram)
    # ==========================================
    # Chuyển sang không gian HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Tính toán histogram với 8x8x8 bins trên vùng quả
    hist = cv2.calcHist([hsv], [0, 1, 2], mask, [8, 8, 8], [0, 180, 0, 256, 0, 256])

    # Chuẩn hóa histogram
    color_features = normalize_histogram(hist)

    # ==========================================
    # ĐẶC TRƯNG 2: HÌNH DÁNG (Hu Moments)
    # ==========================================
    # Tính toán Moments và Hu Moments từ mask vùng quả
    moments = cv2.moments(mask)
    hu_moments = cv2.HuMoments(moments).flatten()

    # Chuẩn hóa logarit cho Hu Moments (vì giá trị gốc thường rất, rất nhỏ)
    # Việc cộng thêm 1e-10 để tránh lỗi log(0)
    shape_features = -np.sign(hu_moments) * np.log10(np.abs(hu_moments) + 1e-10)

    # ==========================================
    # ĐẶC TRƯNG 3: KẾT CẤU (LBP Texture)
    # ==========================================
    # Chuyển ảnh sang ảnh xám để tính LBP
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    lbp = calculate_lbp(gray)

    # Chỉ lấy texture trong vùng quả
    lbp_mask = mask[1:-1, 1:-1]
    if cv2.countNonZero(lbp_mask) > 0:
        texture_values = lbp[lbp_mask > 0]
    else:
        texture_values = lbp.flatten()

    # Histogram 256 bins mô tả phân bố texture bề mặt của quả
    texture_hist, _ = np.histogram(texture_values, bins=LBP_BINS, range=(0, LBP_BINS))
    texture_features = normalize_histogram(texture_hist)

    # ==========================================
    # TỔNG HỢP (Concatenation)
    # ==========================================
    # Nối 512 số màu sắc, 7 số hình dáng và 256 số texture lại với nhau
    # thành vector 775 số
    global_features = np.hstack([color_features, shape_features, texture_features])

    return global_features.astype(np.float32)


# ==== Code test chạy thử (chỉ chạy khi bạn mở trực tiếp file này) ====
if __name__ == "__main__":
    # Đổi thành tên một file ảnh có thật trong máy bạn để test
    test_image = "dataset/query_images/Raspberry 5/r0_7_100.jpg"

    import os

    if os.path.exists(test_image):
        features = extract_features(test_image)
        print("Extraction successful!")
        print(f"Total dimensions of the vector: {len(features)}")
        print(f"Expected dimensions: {FEATURE_DIMENSION}")
        print(f"The 7 shape values: {features[512:519]}")
    else:
        print(f"Test image file not found at: {test_image}")