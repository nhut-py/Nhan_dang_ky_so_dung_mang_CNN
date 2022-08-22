from tkinter import *
import cv2
import numpy as np
from PIL import ImageGrab
from keras.models import load_model
from selenium.webdriver.support.expected_conditions import text_to_be_present_in_element

#Lấy dữ liệu từ file đã training và file img
model = load_model('training/mnists.h5')
image_folder = "img/"

#Tạo form để vẻ với chiêu rộng là 640, chiều cao là 480
root = Tk()
root.resizable(0, 0)
root.title("Dự đoán")
lastx, lasty = None, None
image_number = 0
cv = Canvas(root, width=640, height=480, bg='white')
cv.grid(row=2, column=0, pady=2, sticky=W, columnspan=2)

#Hàm dọn sạch form
def clear_widget():
    global cv
    cv.delete('all')

#Hàm line khi vẻ
def draw_lines(event):
    global lastx, lasty
    x, y = event.x, event.y
    cv.create_line((lastx, lasty, x, y), width=8, fill='black', capstyle=ROUND, smooth=TRUE, splinesteps=12)
    lastx, lasty = x, y


def activate_event(event):
    global lastx, lasty
    cv.bind('<B1-Motion>', draw_lines)
    lastx, lasty = event.x, event.y
cv.bind('<Button-1>', activate_event)
#Hàm lưu ảnh
def Recognize_Digit():
    global image_number
    filename = f'img_{image_number}.png'
    widget = cv
    x = root.winfo_rootx() + widget.winfo_rootx()
    y = root.winfo_rooty() + widget.winfo_rooty()
    x1 = x + widget.winfo_width()
    y1 = y + widget.winfo_height()
    print(x, y, x1, y1)
# Lấy hình ảnh và lưu
    ImageGrab.grab().crop((x, y, x1, y1)).save(image_folder + filename)
    image = cv2.imread(image_folder + filename, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2GRAY)
    ret, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        # make a rectangle box around each curve
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 1)
    # Cắt bỏ chữ số từ hình ảnh tương ứng với các đường bao hiện tại trong vòng lặp for
        digit = th[y:y + h, x:x + w]
    # Thay đổi kích thước chữ số đó thành (18, 18)
        resized_digit = cv2.resize(digit, (18, 18))
    # Chèn chữ số với 5 pixel màu đen (số không) ở mỗi cạnh để 
    # cuối cùng tạo ra hình ảnh của (28, 28)
        padded_digit = np.pad(resized_digit, ((5, 5), (5, 5)), "constant", constant_values=0)

        digit = padded_digit.reshape(1, 28, 28, 1)
        digit = digit / 255.0

        pred = model.predict([digit])[0]
        final_pred = np.argmax(pred)

        data = str(final_pred) + ' ' + str(int(max(pred) * 100)) + '%'

        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 0.5
        color = (255, 0, 0)
        thickness = 1
        cv2.putText(image, data, (x, y - 5), font, fontScale, color, thickness)

    cv2.imshow('img', image)
    cv2.waitKey(0)
# Cấu hình giao diện 2 button
btn_save = Button(text='Nhận diện', fg='Black', activeforeground='Blue', relief=RIDGE, command=Recognize_Digit, activebackground='GhostWhite', bg='GhostWhite', width=15, height=2)
btn_save.grid(row=0, column=0, pady=1, padx=1)
button_clear = Button(text='Dọn dẹp', fg='Black', activeforeground='DarkGray', relief=RIDGE, command=clear_widget,  activebackground='GhostWhite', bg='DeepSkyBlue', width=15, height=2)
button_clear.grid(row=0, column=1, pady=1, padx=1)

root.mainloop()
