"""
GUI Stylesheets for PARTiaL2GLOBAL
==================================
Modern clean light theme with slate and royal blue accents, semantic cards,
and high-contrast typography.
"""

MODERN_STYLESHEET = """
QMainWindow {
    background-color: #f8fafc;
}
QWidget {
    font-family: 'Segoe UI', -apple-system, BlinkMacSystemFont, Arial, sans-serif;
    font-size: 12px;
    color: #0f172a;
}
QGroupBox {
    font-size: 13px;
    font-weight: 700;
    color: #1e3a8a;
    background-color: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    margin-top: 14px;
    padding: 16px 14px 14px 14px;
}
QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 12px;
    padding: 0 8px;
    background-color: #ffffff;
    color: #1e40af;
}
QLabel {
    color: #334155;
}
QLineEdit, QComboBox, QDoubleSpinBox, QSpinBox {
    background-color: #ffffff;
    border: 1px solid #cbd5e1;
    border-radius: 6px;
    padding: 6px 10px;
    color: #0f172a;
    font-size: 12px;
}
QLineEdit:focus, QComboBox:focus, QDoubleSpinBox:focus, QSpinBox:focus {
    border: 2px solid #2563eb;
}
QLineEdit:disabled, QComboBox:disabled, QDoubleSpinBox:disabled, QSpinBox:disabled {
    background-color: #f1f5f9;
    color: #94a3b8;
    border: 1px solid #e2e8f0;
}
QPushButton {
    background-color: #ffffff;
    border: 1px solid #cbd5e1;
    border-radius: 6px;
    padding: 7px 16px;
    font-weight: 600;
    color: #1e293b;
}
QPushButton:hover {
    background-color: #f1f5f9;
    border-color: #94a3b8;
}
QPushButton:pressed {
    background-color: #e2e8f0;
}
QPushButton:disabled {
    background-color: #f8fafc;
    color: #94a3b8;
    border-color: #e2e8f0;
}
QPushButton#primaryButton {
    background-color: #2563eb;
    color: #ffffff;
    border: 1px solid #1d4ed8;
    font-size: 13px;
    font-weight: 700;
    padding: 9px 20px;
}
QPushButton#primaryButton:hover {
    background-color: #1d4ed8;
}
QPushButton#primaryButton:pressed {
    background-color: #1e40af;
}
QPushButton#primaryButton:disabled {
    background-color: #93c5fd;
    border-color: #bfdbfe;
    color: #ffffff;
}
QPushButton#successButton {
    background-color: #16a34a;
    color: #ffffff;
    border: 1px solid #15803d;
    font-weight: 700;
}
QPushButton#successButton:hover {
    background-color: #15803d;
}
QProgressBar {
    border: 1px solid #cbd5e1;
    border-radius: 6px;
    text-align: center;
    background-color: #f1f5f9;
    color: #0f172a;
    font-weight: 600;
    height: 18px;
}
QProgressBar::chunk {
    background-color: #2563eb;
    border-radius: 5px;
}
QPlainTextEdit, QTextEdit {
    background-color: #0f172a;
    color: #f8fafc;
    font-family: 'Consolas', 'Courier New', monospace;
    font-size: 11px;
    border: 1px solid #cbd5e1;
    border-radius: 6px;
    padding: 8px;
}
QTableWidget {
    background-color: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 6px;
    gridline-color: #f1f5f9;
    font-family: 'Consolas', 'Courier New', monospace;
    font-size: 11.5px;
}
QHeaderView::section {
    background-color: #f8fafc;
    color: #475569;
    font-weight: 700;
    padding: 5px 8px;
    border: 1px solid #e2e8f0;
}
QScrollBar:vertical {
    border: none;
    background: #f8fafc;
    width: 8px;
    margin: 0px;
}
QScrollBar::handle:vertical {
    background: #cbd5e1;
    min-height: 20px;
    border-radius: 4px;
}
QScrollBar::handle:vertical:hover {
    background: #94a3b8;
}
"""

METRIC_CARD_STYLE = """
QFrame#metricCard {
    background-color: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    padding: 10px;
}
QLabel#metricTitle {
    color: #64748b;
    font-size: 11px;
    font-weight: 600;
    text-transform: uppercase;
}
QLabel#metricValue {
    color: #0f172a;
    font-size: 20px;
    font-weight: 800;
}
QLabel#metricSub {
    color: #94a3b8;
    font-size: 11px;
}
"""
