import sys
from PyQt6.QtWidgets import QApplication
from gui import GestureApp, create_status_icon

def main():
    try:
        myappid = "mizim.gesture.control.panel.1.0"
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)
    except Exception:
        pass
    
    app = QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(False)
    
    initial_icon = create_status_icon("#ff1744")
    app.setWindowIcon(initial_icon)

    window = GestureApp()
    window.show()
    
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
