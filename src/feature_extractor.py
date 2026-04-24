import cv2
import numpy as np

def extract_features(image_path):
    """
    Trích xuất kết hợp Màu sắc (Color Histogram) và Hình dáng (Hu Moments).
    Trả về một vector đặc trưng duy nhất.
    """
    # Đọc ảnh
    image = cv2.imread(image_path)
    if image is None:
        return None

    # ==========================================
    # ĐẶC TRƯNG 1: MÀU SẮC (Color Histogram)
    # ==========================================
    # Chuyển sang không gian HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # Tính toán histogram với 8x8x8 bins
    hist = cv2.calcHist([hsv], [0, 1, 2], None, [8, 8, 8], [0, 180, 0, 256, 0, 256])
    # Chuẩn hóa histogram
    cv2.normalize(hist, hist)
    # Làm phẳng thành vector 1D (512 chiều)
    color_features = hist.flatten()

    # ==========================================
    # ĐẶC TRƯNG 2: HÌNH DÁNG (Hu Moments)
    # ==========================================
    # Chuyển ảnh sang ảnh xám (Grayscale)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Vì ảnh Fruits 360 có nền trắng, ta dùng Threshold để tách quả ra khỏi nền.
    # Mọi điểm ảnh tối hơn màu trắng (< 240) sẽ biến thành số 1 (màu trắng), nền biến thành số 0 (màu đen).
    _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
    
    # Tính toán Moments và Hu Moments từ ảnh đã tách nền
    moments = cv2.moments(thresh)
    hu_moments = cv2.HuMoments(moments).flatten()
    
    # Chuẩn hóa logarit cho Hu Moments (vì giá trị gốc thường rất, rất nhỏ)
    # Việc cộng thêm 1e-10 để tránh lỗi log(0)
    hu_moments = -np.sign(hu_moments) * np.log10(np.abs(hu_moments) + 1e-10)

    # ==========================================
    # TỔNG HỢP (Concatenation)
    # ==========================================
    # Nối 512 số màu sắc và 7 số hình dáng lại với nhau thành vector 519 số
    global_features = np.hstack([color_features, hu_moments])

    return global_features

# ==== Code test chạy thử (chỉ chạy khi bạn mở trực tiếp file này) ====
if __name__ == "__main__":
    # Đổi thành tên một file ảnh có thật trong máy bạn để test
    test_image = "dataset/query_images/Raspberry 5/r0_7_100.jpg " 
    
    import os
    if os.path.exists(test_image):
        features = extract_features(test_image)
        print("Extraction successful!")
        print(f"Total dimensions of the vector:{len(features)}")
        print(f"The 7 shape values: {features[-7:]}")
    else:
        print(f"Test image file not found at: {test_image}")