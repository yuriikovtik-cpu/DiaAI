import sys
from PySide6.QtWidgets import ( QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QTextEdit, QStackedWidget, QFrame)
from PySide6.QtCore import Qt, QPropertyAnimation, QEasingCurve, QRect
from PySide6.QtGui import QPixmap, QLinearGradient, QPalette, QColor, QBrush
import neuralNetwork as neurn

def send_to_ai(text: str, old_text: str,name: str,number: str) -> str:
    return f"{name} ми отримали ваше звернення: '{old_text}'\nзверніться до:  {text}\n{number}"
class ResponseScreen(QWidget):
    def __init__(self, parent, response_text, return_callback, user_name):
        super().__init__()
        self.parent = parent
        self.return_cb = return_callback
        self.user_name = user_name
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.Dialog)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.resize(parent.size())  # <-- робимо повноекранним
        self.init_ui(response_text)
        self.show_animation()

    def init_ui(self, response_text):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0,0,0,0)
        layout.setAlignment(Qt.AlignCenter)

        self.card = QFrame()
        self.card.setStyleSheet(
            "background-color:#1e1e1e; border-radius:20px;"
        )
        self.card.setFixedSize(400, 400)
        card_layout = QVBoxLayout(self.card)
        card_layout.setContentsMargins(20,20,20,20)
        card_layout.setSpacing(12)

        self.lbl_response = QLabel(response_text)
        self.lbl_response.setWordWrap(True)
        self.lbl_response.setStyleSheet("color:white; font-size:16px;")
        card_layout.addWidget(self.lbl_response)

        self.txt_new = QTextEdit()
        self.txt_new.setPlaceholderText("Маєте ще одну скаргу? Введіть її!")
        self.txt_new.setStyleSheet("background:#d0f0c0; color:black; border-radius:12px;")
        self.txt_new.setFixedHeight(100)
        card_layout.addWidget(self.txt_new)

        btn_layout = QHBoxLayout()
        self.btn_send = QPushButton("Надіслати")
        self.btn_close = QPushButton("Закрити")
        self.btn_send.setStyleSheet(self.button_style())
        self.btn_close.setStyleSheet(self.button_style())
        self.btn_send.clicked.connect(self.on_send)
        self.btn_close.clicked.connect(self.on_close)
        btn_layout.addWidget(self.btn_send)
        btn_layout.addWidget(self.btn_close)
        card_layout.addLayout(btn_layout)

        layout.addWidget(self.card)

        gradient = QLinearGradient(0,0,0,self.height())
        gradient.setColorAt(0.0, QColor(100, 181, 246, 180))
        gradient.setColorAt(1.0, QColor(105, 240, 174, 180))
        palette = QPalette()
        palette.setBrush(QPalette.Window, QBrush(gradient))
        self.setPalette(palette)
        self.setAutoFillBackground(True)

    def show_animation(self):
        self.anim = QPropertyAnimation(self.card, b"geometry")
        # стартова геометрія: маленька точка в центрі екрану
        start_rect = QRect(self.width()//2, self.height()//2, 0, 0)
        # кінцева геометрія: картка 400x400 по центру
        end_rect = QRect((self.width()-400)//2, (self.height()-400)//2, 400, 400)
        self.card.setGeometry(start_rect)
        self.anim.setStartValue(start_rect)
        self.anim.setEndValue(end_rect)
        self.anim.setDuration(300)
        self.anim.setEasingCurve(QEasingCurve.OutBack)
        self.anim.start()

    def button_style(self):
        return """
            QPushButton {
                background-color: #000000;
                color: white;
                border-radius: 12px;
                padding: 10px;
                font-size:16px;
            }
            QPushButton:pressed {
                background-color: #333333;
            }
        """

    def on_send(self):
        text = self.txt_new.toPlainText().strip()
        if not text:
            return
        service = neurn.services[neurn.classify_query(text)]
        number = neurn.numbers[neurn.classify_query(text)]
        response = send_to_ai(service, text, self.user_name, number)
        self.lbl_response.setText(response)
        self.txt_new.clear()

    def on_close(self):
        self.setParent(None)
        self.deleteLater()
        self.return_cb()  # повернення на головну сторінку

class MainScreen(QWidget):
    def __init__(self, switch_callback):
        super().__init__()
        self.switch_cb = switch_callback
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20,20,20,20)
        layout.setSpacing(20)

        # Заголовок
        title = QLabel("Дія - Профіль")
        title.setStyleSheet("font-size:28px; color:black; background:#d0f0c0; border-radius:12px; padding:5px;")
        layout.addWidget(title)

        # Картка з аватаркою та ID
        card = QFrame()
        card.setStyleSheet("background:#333333; border-radius:16px;")
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(20,20,20,20)
        card_layout.setSpacing(10)

        avatar = QLabel()
        try:
            pix = QPixmap("avatar.png")
            if not pix.isNull():
                avatar.setPixmap(pix.scaledToHeight(140, Qt.SmoothTransformation))
        except Exception:
            pass
        avatar.setAlignment(Qt.AlignCenter)
        card_layout.addWidget(avatar)

        self.name = QLabel("Вишня Рудольф Григорович")
        self.name.setStyleSheet("font-size:20px; color:white;")
        self.name.setAlignment(Qt.AlignCenter)
        card_layout.addWidget(self.name)

        id_lbl = QLabel("ID: 44 123 446")
        id_lbl.setStyleSheet("color:white; font-size:16px;")
        id_lbl.setAlignment(Qt.AlignCenter)
        card_layout.addWidget(id_lbl)

        layout.addWidget(card)
        layout.addStretch(1)

        btn = QPushButton("Перейти до скарги")
        btn.setFixedHeight(48)
        btn.setStyleSheet(self.button_style())
        btn.clicked.connect(self.switch_cb)
        layout.addWidget(btn)

        self.setStyleSheet("background: qlineargradient(x1:0,y1:0,x2:0,y2:1, stop:0 #64B5F6, stop:1 #69F0AE);")

    def button_style(self):
        return """
            QPushButton {
                background-color: #000000;
                color: white;
                border-radius: 12px;
                padding: 10px;
                font-size:16px;
            }
            QPushButton:pressed {
                background-color: #333333;
            }
        """

class ComplaintScreen(QWidget):
    def __init__(self, open_popup_cb, main_screen):
        super().__init__()
        self.open_popup_cb = open_popup_cb
        self.main_screen = main_screen
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20,20,20,20)
        layout.setSpacing(15)

        title = QLabel("Введіть свою скаргу")
        title.setStyleSheet("font-size:20px; color:black; background:#d0f0c0; border-radius:12px; padding:5px;")
        layout.addWidget(title)

        self.inputText = QTextEdit()
        self.inputText.setPlaceholderText("Опишіть проблему...")
        self.inputText.setStyleSheet("background:#d0f0c0; color:black; border-radius:12px;")
        layout.addWidget(self.inputText, stretch=1)

        btn = QPushButton("Надіслати")
        btn.setFixedHeight(44)
        btn.setStyleSheet("""
            QPushButton {
                background-color: #000000;
                color: white;
                border-radius: 12px;
                padding: 10px;
                font-size:16px;
            }
            QPushButton:pressed {
                background-color: #333333;
            }
        """)
        btn.clicked.connect(self.on_send)
        layout.addWidget(btn)

        self.setStyleSheet("background: qlineargradient(x1:0,y1:0,x2:0,y2:1, stop:0 #64B5F6, stop:1 #69F0AE);")

    def on_send(self):
        text = self.inputText.toPlainText().strip()
        if not text:
            return
        service = neurn.services[neurn.classify_query(text)]
        user_name = self.main_screen.name.text()
        number = neurn.numbers[neurn.classify_query(text)]
        response = send_to_ai(service, text, user_name, number)
        self.open_popup_cb(response)
        self.inputText.clear()

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.resize(540,900)
        self.setWindowTitle("Додаток звернень")
        self.stack = QStackedWidget()
        self.current_index = 0

        self.main_screen = MainScreen(self.show_complaint)
        self.complaint_screen = ComplaintScreen(self.show_response, self.main_screen)
        self.stack.addWidget(self.main_screen)
        self.stack.addWidget(self.complaint_screen)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0,0,0,0)
        layout.addWidget(self.stack)

        self.dots_layout = QHBoxLayout()
        self.dots_layout.setSpacing(10)
        self.dots_layout.setAlignment(Qt.AlignCenter)
        self.dots = [QLabel("●"), QLabel("○")]
        for dot in self.dots:
            dot.setStyleSheet("font-size:18px; color:black;")
            self.dots_layout.addWidget(dot)
        layout.addLayout(self.dots_layout)
        self.update_dots()

        self.start_pos = None

    def update_dots(self):
        for i,dot in enumerate(self.dots):
            dot.setText("●" if i==self.current_index else "○")

    def show_complaint(self):
        self.current_index=1
        self.stack.setCurrentIndex(1)
        self.update_dots()

    def show_response(self, response_text):
        user_name = self.main_screen.name.text()
        self.popup = ResponseScreen(self, response_text, self.back_to_main, user_name)
        self.popup.show()

    def back_to_main(self):
        self.current_index=0
        self.stack.setCurrentIndex(0)
        self.update_dots()

    def mousePressEvent(self,event):
        if event.button()==Qt.LeftButton:
            self.start_pos=event.pos()

    def mouseReleaseEvent(self,event):
        if self.start_pos is None:
            return
        end_pos=event.pos()
        dx=end_pos.x()-self.start_pos.x()
        if dx>50 and self.current_index>0:
            self.current_index-=1
        elif dx<-50 and self.current_index<self.stack.count()-1:
            self.current_index+=1
        self.stack.setCurrentIndex(self.current_index)
        self.update_dots()
        self.start_pos=None

if __name__=="__main__":
    app=QApplication(sys.argv)
    win=MainWindow()
    win.show()
    sys.exit(app.exec())
