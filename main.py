from kivy.uix.image import Image
from kivy.uix.togglebutton import ToggleButton
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.config import Config
from kivy.graphics import Color, RoundedRectangle
import cv2
import os
from collections import Counter
from datetime import datetime
import numpy as np

# Импорты KivyMD
from kivymd.app import MDApp
from kivymd.uix.card import MDCard
from kivymd.uix.button import MDRaisedButton, MDFillRoundFlatButton, MDFlatButton
from kivymd.uix.label import MDLabel
from kivymd.uix.scrollview import MDScrollView
from kivymd.uix.toolbar import MDTopAppBar
from kivymd.uix.list import MDList, OneLineListItem
from kivymd.uix.dialog import MDDialog
from kivymd.uix.boxlayout import MDBoxLayout

# Улучшение детекции
last_detections = []
detection_history = []
FRAME_SKIP = 3  # Обрабатываем каждый 3-й кадр
current_frame_count = 0

# Настройки конфигурации
Config.set('input', 'mouse', 'mouse,multitouch_on_demand')
Config.set('graphics', 'multisamples', '0')
os.environ['KIVY_NO_CONSOLELOG'] = '1'

# ===== КОНФИГУРАЦИЯ МОДЕЛИ YOLO =====
# ИЗМЕНЕНО: Используем ONNX вместо .pt для совместимости с APK
model_path = 'yolov8n.onnx'

if not os.path.exists(model_path):
    print(f"ONNX модель не найдена: {model_path}")
    print("Создай ONNX модель запустив: python converter.py")
    # Создаем заглушку чтобы приложение не падало
    onnx_net = None
else:
    print(f"ONNX модель найдена: {model_path}")
    # Загрузка ONNX модели
    onnx_net = cv2.dnn.readNetFromONNX(model_path)
    print("ONNX модель загружена")

# Классы YOLO COCO (стандартные для yolov8n)
yolo_classes = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
    'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
    'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
    'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
    'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
    'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

def detect_objects(frame):
    """Обнаружение объектов с улучшенной фильтрацией дубликатов"""
    global last_detections, detection_history, current_frame_count
    
    if onnx_net is None:
        return []
    
    # ПРОПУСКАЕМ КАДРЫ ДЛЯ СНИЖЕНИЯ НАГРУЗКИ
    current_frame_count += 1
    if current_frame_count % FRAME_SKIP != 0:
        return last_detections
    
    try:
        blob = cv2.dnn.blobFromImage(frame, 1/255.0, (640, 640), swapRB=True, crop=False)
        onnx_net.setInput(blob)
        outputs = onnx_net.forward()
        predictions = outputs[0]
        
        current_detections = []
        all_boxes = []
        all_confidences = []
        all_class_ids = []
        
        # Собираем все детекции
        for i in range(predictions.shape[1]):
            prediction = predictions[:, i]
            x_center, y_center, width, height = prediction[0:4]
            scores = prediction[4:84]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            
            # РАЗНЫЕ ПОРОГИ ДЛЯ РАЗНЫХ КЛАССОВ
            class_name = yolo_classes[class_id]
            if class_name == 'person':
                min_confidence = 0.5  # Повышенный порог для людей
            else:
                min_confidence = 0.4
            
            if confidence > min_confidence:
                scale_x = frame.shape[1] / 640
                scale_y = frame.shape[0] / 640
                
                x1 = int((x_center - width/2) * scale_x)
                y1 = int((y_center - height/2) * scale_y)
                x2 = int((x_center + width/2) * scale_x)
                y2 = int((y_center + height/2) * scale_y)
                
                x1 = max(0, min(x1, frame.shape[1]))
                y1 = max(0, min(y1, frame.shape[0]))
                x2 = max(0, min(x2, frame.shape[1]))
                y2 = max(0, min(y2, frame.shape[0]))
                
                if x2 > x1 and y2 > y1 and class_id < len(yolo_classes):
                    # Фильтр по минимальному размеру
                    bbox_width = x2 - x1
                    bbox_height = y2 - y1
                    min_size = 50 if class_name == 'person' else 30
                    
                    if bbox_width > min_size and bbox_height > min_size:
                        all_boxes.append([x1, y1, x2, y2])
                        all_confidences.append(float(confidence))
                        all_class_ids.append(class_id)
        
        # ПРИМЕНЯЕМ NMS ДЛЯ КАЖДОГО КЛАССА ОТДЕЛЬНО
        if all_boxes:
            # Разделяем по классам
            person_boxes = []
            person_confidences = []
            person_indices = []
            
            other_boxes = []
            other_confidences = []
            other_indices = []
            
            for i, class_id in enumerate(all_class_ids):
                if yolo_classes[class_id] == 'person':
                    person_boxes.append(all_boxes[i])
                    person_confidences.append(all_confidences[i])
                    person_indices.append(i)
                else:
                    other_boxes.append(all_boxes[i])
                    other_confidences.append(all_confidences[i])
                    other_indices.append(i)
            
            # NMS для людей - более агрессивный
            if person_boxes:
                person_indices_nms = cv2.dnn.NMSBoxes(
                    person_boxes, person_confidences, 
                    score_threshold=0.5,
                    nms_threshold=0.4  # Более агрессивный для людей
                )
                
                if len(person_indices_nms) > 0:
                    for idx in person_indices_nms.flatten():
                        original_idx = person_indices[idx]
                        current_detections.append({
                            'class': 'person',
                            'confidence': all_confidences[original_idx],
                            'bbox': all_boxes[original_idx]
                        })
            
            # NMS для остальных объектов
            if other_boxes:
                other_indices_nms = cv2.dnn.NMSBoxes(
                    other_boxes, other_confidences,
                    score_threshold=0.4,
                    nms_threshold=0.3
                )
                
                if len(other_indices_nms) > 0:
                    for idx in other_indices_nms.flatten():
                        original_idx = other_indices[idx]
                        current_detections.append({
                            'class': yolo_classes[all_class_ids[original_idx]],
                            'confidence': all_confidences[original_idx],
                            'bbox': all_boxes[original_idx]
                        })
        
        # ОБНОВЛЯЕМ ИСТОРИЮ ДЕТЕКЦИЙ
        last_detections = current_detections
        
        return current_detections
        
    except Exception as e:
        print(f"Ошибка в detect_objects: {e}")
        return last_detections
    
def draw_boxes(frame, detected_objects):
    """Улучшенная отрисовка bounding boxes"""
    for obj in detected_objects:
        x1, y1, x2, y2 = obj['bbox']
        confidence = obj['confidence']
        class_name = obj['class']
        
        # РАЗНЫЕ ЦВЕТА ДЛЯ РАЗНЫХ КЛАССОВ
        if class_name in ['cup', 'bottle', 'wine glass']:
            color = (255, 0, 0)  # Синий для посуды
            thickness = 3
        elif class_name in ['cell phone', 'laptop', 'mouse']:
            color = (0, 0, 255)  # Красный для электроники
            thickness = 3
        elif class_name == 'person':
            color = (0, 255, 0)  # Зеленый для людей
            thickness = 2
        else:
            color = (255, 255, 0)  # Голубой для остального
            thickness = 2
        
        # Отрисовка рамки
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
        
        # Фон для текста
        text = f"{class_name} {confidence*100:.0f}%"
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        cv2.rectangle(frame, (x1, y1 - text_size[1] - 10), 
                     (x1 + text_size[0], y1), color, -1)
        
        # Текст
        cv2.putText(frame, text, (x1, y1 - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    return frame, detected_objects


class MaterialToggleButton(ToggleButton):
    """Кастомная кнопка-переключатель в стиле Material Design"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.background_color = (0, 0, 0, 0)
        self.background_normal = ''
        self.background_down = ''
        self.color = (1, 1, 1, 1)
        self.font_size = '14sp'
        self.bold = True
        self.group = 'snapshot_type'

        self.bind(state=self.update_graphics)
        self.bind(pos=self.update_graphics, size=self.update_graphics)

    def update_graphics(self, *args):
        """Обновление графического представления кнопки"""
        self.canvas.before.clear()
        with self.canvas.before:
            if self.state == 'down':
                # Активное состояние
                Color(0.2, 0.6, 0.9, 1)
                RoundedRectangle(pos=self.pos, size=self.size, radius=[20])
                Color(1, 1, 1, 1)
            else:
                # Неактивное состояние
                Color(0.7, 0.7, 0.7, 1)
                RoundedRectangle(pos=self.pos, size=self.size, radius=[20])
                Color(0.3, 0.3, 0.3, 1)


class CameraCard(MDCard):
    """Карточка для отображения видео с камеры"""
    pass


class ObjectsCard(MDCard):
    """Карточка для отображения обнаруженных объектов"""
    pass


class ControlCard(MDCard):
    """Карточка с элементами управления"""
    pass


class DetectionListItem(OneLineListItem):
    """Элемент списка для отображения обнаруженных объектов"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.text_color = (1, 1, 1, 1)
        self.bg_color = (0.9, 0.95, 1, 1)


class MainLayout(MDBoxLayout):
    """Главный layout приложения"""

    def __init__(self, **kwargs):
        super().__init__(orientation='vertical', **kwargs)
        self.spacing = "8dp"
        self.padding = "8dp"

        # Инициализация компонентов интерфейса
        self._init_ui()

        # Инициализация переменных состояния
        self._init_variables()

        # Проверка доступности камер
        self.check_cameras_availability()

    def _init_ui(self):
        """Инициализация пользовательского интерфейса"""
        self._create_top_app_bar()
        self._create_main_content()
        self._create_control_panel()
        self._create_status_bar()

    def _init_variables(self):
        """Инициализация переменных состояния"""
        self.capture = None
        self.event = None
        self.current_frame = None
        self.current_camera_index = 0
        self.available_cameras = [0, 1]
        self.detected_objects_history = []
        self.last_frame = None
        self.save_with_detection = True
        self.last_objects_signature = ""
        self.last_log_time = None
        self.info_dialog = None
        self.camera_info_dialog = None

    def _create_top_app_bar(self):
        """Создание верхней панели приложения"""
        self.top_app_bar = MDTopAppBar(
            title="YOLO Object Detection",
            elevation=4,
            md_bg_color=(0.2, 0.3, 0.5, 1),
            specific_text_color=(1, 1, 1, 1),
            left_action_items=[["camera", lambda x: self.show_camera_info_action()]],
            right_action_items=[["information", lambda x: self.show_app_info_action()]]
        )
        self.add_widget(self.top_app_bar)

    def _create_main_content(self):
        """Создание основной области контента"""
        self.main_content = MDBoxLayout(orientation='vertical', spacing="8dp", size_hint=(1, 1))

        self._create_camera_info_section()
        self._create_video_objects_section()
        self._create_settings_section()

        self.add_widget(self.main_content)

    def _create_camera_info_section(self):
        """Создание секции информации о камере"""
        self.camera_info_card = MDCard(
            orientation="vertical", padding="12dp", size_hint_y=None, height="70dp",
            elevation=3, md_bg_color=(0.2, 0.4, 0.6, 1)
        )

        self.camera_info_layout = MDBoxLayout(orientation="horizontal")

        # Метка статуса камеры
        self.camera_status_label = MDLabel(
            text="Камера: ОСНОВНАЯ", theme_text_color="Custom", text_color=(1, 1, 1, 1),
            bold=True, size_hint_x=0.6
        )

        # Кнопка переключения камеры
        self.camera_switch_btn = MDFillRoundFlatButton(
            text="ПЕРЕКЛЮЧИТЬ", size_hint_x=0.4, size_hint_y=None, height="35dp",
            md_bg_color=(0.3, 0.5, 0.7, 1)
        )
        self.camera_switch_btn.bind(on_press=self.switch_camera)

        self.camera_info_layout.add_widget(self.camera_status_label)
        self.camera_info_layout.add_widget(self.camera_switch_btn)
        self.camera_info_card.add_widget(self.camera_info_layout)
        self.main_content.add_widget(self.camera_info_card)

    def _create_video_objects_section(self):
        """Создание секции видео и обнаруженных объектов"""
        self.video_objects_layout = MDBoxLayout(orientation='horizontal', spacing="8dp", size_hint=(1, 1))

        self._create_video_display()
        self._create_objects_panel()

        self.main_content.add_widget(self.video_objects_layout)

    def _create_video_display(self):
        """Создание области отображения видео"""
        self.video_card = CameraCard(
            orientation="vertical", padding="8dp", elevation=4,
            md_bg_color=(0.1, 0.1, 0.1, 1), size_hint_x=0.7
        )

        self.video_display = Image(size_hint=(1, 1), fit_mode='contain')
        self.video_card.add_widget(self.video_display)
        self.video_objects_layout.add_widget(self.video_card)

    def _create_objects_panel(self):
        """Создание панели обнаруженных объектов"""
        self.objects_card = ObjectsCard(
            orientation="vertical", padding="8dp", elevation=4, size_hint_x=0.3
        )

        # Заголовок панели объектов
        self.objects_header = MDBoxLayout(orientation="vertical", size_hint_y=None, height="60dp")

        self.objects_label = MDLabel(
            text="ОБНАРУЖЕННЫЕ", theme_text_color="Primary", bold=True,
            halign="center", size_hint_y=None, height="25dp"
        )

        self.objects_label2 = MDLabel(
            text="ОБЪЕКТЫ", theme_text_color="Primary", bold=True,
            halign="center", size_hint_y=None, height="25dp"
        )

        self.objects_count_label = MDLabel(
            text="0 объектов", theme_text_color="Secondary",
            halign="center", size_hint_y=None, height="20dp"
        )

        self.objects_header.add_widget(self.objects_label)
        self.objects_header.add_widget(self.objects_label2)
        self.objects_header.add_widget(self.objects_count_label)
        self.objects_card.add_widget(self.objects_header)

        # Список объектов со скроллом
        self.scroll_view = MDScrollView(
            size_hint=(1, 1), do_scroll_x=False, do_scroll_y=True,
            bar_width='8dp', bar_color=(0.5, 0.5, 0.5, 0.5)
        )
        self.objects_list = MDList()
        self.scroll_view.add_widget(self.objects_list)
        self.objects_card.add_widget(self.scroll_view)

        self.video_objects_layout.add_widget(self.objects_card)

    def _create_settings_section(self):
        """Создание секции настроек снимков"""
        self.settings_card = MDCard(
            orientation="vertical", padding="12dp", size_hint_y=None,
            height="100dp", elevation=3
        )

        self.settings_label = MDLabel(
            text="НАСТРОЙКИ СНИМКОВ", theme_text_color="Primary", bold=True,
            halign="center", size_hint_y=None, height="25dp"
        )
        self.settings_card.add_widget(self.settings_label)

        # Кнопки переключения режимов снимков
        self.switch_layout = MDBoxLayout(
            orientation="horizontal", spacing="15dp", size_hint_y=None,
            height="40dp", padding="10dp"
        )

        self.clean_snapshot_btn = MaterialToggleButton(text="ЧИСТЫЕ СНИМКИ", size_hint=(0.48, 1))
        self.clean_snapshot_btn.bind(on_press=self.on_snapshot_type_changed)

        self.detection_snapshot_btn = MaterialToggleButton(text="С ДЕТЕКЦИЕЙ", size_hint=(0.48, 1))
        self.detection_snapshot_btn.bind(on_press=self.on_snapshot_type_changed)

        # Установка режима детекции по умолчанию
        self.detection_snapshot_btn.state = 'down'
        self.save_with_detection = True

        self.switch_layout.add_widget(self.clean_snapshot_btn)
        self.switch_layout.add_widget(self.detection_snapshot_btn)
        self.settings_card.add_widget(self.switch_layout)

        self.snapshot_type_label = MDLabel(
            text="Текущий режим: С ДЕТЕКЦИЕЙ", theme_text_color="Secondary",
            halign="center", size_hint_y=None, height="20dp"
        )
        self.settings_card.add_widget(self.snapshot_type_label)

        self.main_content.add_widget(self.settings_card)

    def _create_control_panel(self):
        """Создание панели управления"""
        self.control_card = ControlCard(
            orientation="horizontal", padding="15dp", spacing="10dp",
            size_hint_y=None, height="80dp", elevation=4,
            md_bg_color=(0.9, 0.9, 0.9, 1)
        )

        # Кнопки управления
        self.start_button = MDRaisedButton(
            text="СТАРТ", size_hint_x=0.25, md_bg_color=(0.2, 0.7, 0.3, 1)
        )
        self.start_button.bind(on_press=self.start_video)

        self.stop_button = MDRaisedButton(
            text="СТОП", size_hint_x=0.25, md_bg_color=(0.8, 0.2, 0.2, 1)
        )
        self.stop_button.bind(on_press=self.stop_video)

        self.snapshot_button = MDRaisedButton(
            text="СНИМОК", size_hint_x=0.25, md_bg_color=(0.9, 0.6, 0.1, 1)
        )
        self.snapshot_button.bind(on_press=self.save_snapshot)

        self.info_button = MDRaisedButton(
            text="ИНФО", size_hint_x=0.25, md_bg_color=(0.3, 0.4, 0.8, 1)
        )
        self.info_button.bind(on_press=self.show_camera_info)

        self.control_card.add_widget(self.start_button)
        self.control_card.add_widget(self.stop_button)
        self.control_card.add_widget(self.snapshot_button)
        self.control_card.add_widget(self.info_button)

        self.add_widget(self.control_card)

    def _create_status_bar(self):
        """Создание строки статуса"""
        self.status_bar = MDBoxLayout(
            orientation="horizontal", size_hint_y=None, height="35dp",
            padding="8dp", md_bg_color=(0.2, 0.2, 0.3, 1)
        )

        self.status_indicator = MDLabel(
            text="СИСТЕМА ГОТОВА К РАБОТЕ", theme_text_color="Custom",
            text_color=(1, 1, 1, 1), bold=True, halign="center"
        )

        self.status_bar.add_widget(self.status_indicator)
        self.add_widget(self.status_bar)

    def check_cameras_availability(self):
        """Проверка доступности камер и обновление интерфейса"""
        print("Проверка доступности камер...")
        available_cameras = []

        for i in [0, 1]:
            try:
                cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
                if cap.isOpened():
                    ret, frame = cap.read()
                    if ret:
                        available_cameras.append(i)
                        print(f"Камера {i} доступна")
                    cap.release()
                else:
                    print(f"Камера {i} недоступна")
            except Exception as e:
                print(f"Ошибка при проверке камеры {i}: {e}")

        self.available_cameras = available_cameras
        self._update_camera_ui()

    def _update_camera_ui(self):
        """Обновление интерфейса в зависимости от доступности камер"""
        if len(self.available_cameras) == 0:
            self.camera_switch_btn.md_bg_color = (0.8, 0.2, 0.2, 1)
            self.camera_status_label.text = "Камера: НЕДОСТУПНА"
            self.status_indicator.text = "ОШИБКА: КАМЕРЫ НЕДОСТУПНЫ"
            self.status_indicator.text_color = (1, 0.3, 0.3, 1)
        elif len(self.available_cameras) == 1:
            self.camera_switch_btn.md_bg_color = (1, 0.6, 0.1, 1)
            self.current_camera_index = self.available_cameras[0]
            camera_name = "ОСНОВНАЯ" if self.current_camera_index == 0 else "ФРОНТАЛЬНАЯ"
            self.camera_status_label.text = f"Камера: {camera_name}"
            mode_text = "ЧИСТЫЕ" if not self.save_with_detection else "С ДЕТЕКЦИЕЙ"
            self.status_indicator.text = f"1 КАМЕРА | СНИМКИ: {mode_text}"
            self.status_indicator.text_color = (1, 0.8, 0.3, 1)
        else:
            self.camera_switch_btn.md_bg_color = (0.3, 0.5, 0.7, 1)
            self.camera_status_label.text = "Камера: ОСНОВНАЯ"
            mode_text = "ЧИСТЫЕ" if not self.save_with_detection else "С ДЕТЕКЦИЕЙ"
            self.status_indicator.text = f"2 КАМЕРЫ | СНИМКИ: {mode_text}"
            self.status_indicator.text_color = (0.3, 0.8, 0.3, 1)

    def show_camera_info_action(self):
        """Обработчик кнопки информации о камере"""
        self.show_camera_info(None)

    def show_app_info_action(self):
        """Обработчик кнопки информации о приложении"""
        self.show_app_info()

    def on_snapshot_type_changed(self, *_):
        """Обработчик изменения типа снимка"""
        if self.clean_snapshot_btn.state == 'down':
            self.save_with_detection = False
            self.snapshot_type_label.text = "Текущий режим: ЧИСТЫЕ СНИМКИ"
            self.status_indicator.text = "РЕЖИМ: ЧИСТЫЕ СНИМКИ"
        elif self.detection_snapshot_btn.state == 'down':
            self.save_with_detection = True
            self.snapshot_type_label.text = "Текущий режим: С ДЕТЕКЦИЕЙ"
            self.status_indicator.text = "РЕЖИМ: СНИМКИ С ДЕТЕКЦИЕЙ"

    def show_app_info(self):
        """Отображение информации о приложении"""
        if not self.info_dialog:
            self.info_dialog = MDDialog(
                title="О приложении",
                text="YOLO Object Detection v2.0\n\n"
                     "Система компьютерного зрения на основе YOLOv8\n"
                     "для обнаружения объектов в реальном времени.\n\n"
                     "Разработано Aravakesh -Elmat-\n"
                     "и курсов УИИ.",
                size_hint=(0.8, 0.4),
                buttons=[
                    MDFlatButton(
                        text="OK", theme_text_color="Custom",
                        text_color=(0.2, 0.5, 0.8, 1),
                        on_release=lambda x: self.info_dialog.dismiss()
                    ),
                ],
            )
        self.info_dialog.open()

    def switch_camera(self, *_):
        """Переключение между камерами"""
        if not self.available_cameras:
            print("Нет доступных камер для переключения")
            self.status_indicator.text = "НЕТ ДОСТУПНЫХ КАМЕР"
            self.status_indicator.text_color = (1, 0.3, 0.3, 1)
            return

        if len(self.available_cameras) == 1:
            print("Доступна только одна камера, переключение невозможно")
            self.status_indicator.text = "ПЕРЕКЛЮЧЕНИЕ НЕВОЗМОЖНО"
            self.status_indicator.text_color = (1, 0.8, 0.3, 1)
            Clock.schedule_once(lambda dt: self.check_cameras_availability(), 2)
            return

        # Остановка текущего видео
        was_playing = self.event is not None
        if was_playing:
            self.stop_video(None)

        # Переключение на следующую камеру
        current_index = self.available_cameras.index(self.current_camera_index)
        next_index = (current_index + 1) % len(self.available_cameras)
        self.current_camera_index = self.available_cameras[next_index]

        # Обновление интерфейса
        camera_name = "ОСНОВНАЯ" if self.current_camera_index == 0 else "ФРОНТАЛЬНАЯ"
        self.camera_status_label.text = f"Камера: {camera_name}"
        print(f"Переключено на камеру: {camera_name}")

        # Запуск видео с новой камеры
        if was_playing:
            self.start_video(None)

    def show_camera_info(self, *_):
        """Отображение информации о текущей камере"""
        if self.capture is not None and self.capture.isOpened():
            width = self.capture.get(cv2.CAP_PROP_FRAME_WIDTH)
            height = self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
            fps = self.capture.get(cv2.CAP_PROP_FPS)

            info_text = (f"Информация о камере:\n\n"
                         f"• Разрешение: {int(width)}x{int(height)}\n"
                         f"• FPS: {fps:.1f}\n"
                         f"• Индекс: {self.current_camera_index}\n"
                         f"• Тип снимков: {'С ДЕТЕКЦИЕЙ' if self.save_with_detection else 'ЧИСТЫЕ'}\n"
                         f"• Статус: {'АКТИВНА' if self.event else 'ОСТАНОВЛЕНА'}")

            if not self.camera_info_dialog:
                self.camera_info_dialog = MDDialog(
                    title="Информация о камере",
                    text=info_text,
                    size_hint=(0.8, 0.4),
                    buttons=[
                        MDFlatButton(
                            text="OK", theme_text_color="Custom",
                            text_color=(0.2, 0.5, 0.8, 1),
                            on_release=lambda x: self.camera_info_dialog.dismiss()
                        ),
                    ],
                )
            else:
                self.camera_info_dialog.text = info_text
            self.camera_info_dialog.open()

            mode_text = "ЧИСТЫЕ" if not self.save_with_detection else "С ДЕТЕКЦИЕЙ"
            status_text = "АКТИВНА" if self.event else "ОСТАНОВЛЕНА"
            self.status_indicator.text = f"ИНФО: {int(width)}x{int(height)} | СНИМКИ: {mode_text} | {status_text}"
            self.status_indicator.text_color = (0.2, 0.6, 0.9, 1)
        else:
            self.status_indicator.text = "КАМЕРА НЕ АКТИВНА"
            self.status_indicator.text_color = (1, 0.3, 0.3, 1)
            Clock.schedule_once(lambda dt: self.check_cameras_availability(), 2)

    def start_video(self, *_):
        """Запуск видеопотока"""
        print(f"Запуск видеопотока (камера {self.current_camera_index})")

        if self.capture is not None:
            self.capture.release()

        if self.current_camera_index not in self.available_cameras:
            print(f"Камера {self.current_camera_index} недоступна")
            self.status_indicator.text = "ОШИБКА: КАМЕРА НЕДОСТУПНА"
            self.status_indicator.text_color = (1, 0.3, 0.3, 1)
            self.check_cameras_availability()
            return

        camera_index = self.current_camera_index
        self.capture = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)

        if not self.capture.isOpened():
            print(f"Не удалось открыть камеру {camera_index}")
            self.status_indicator.text = "ОШИБКА ОТКРЫТИЯ КАМЕРЫ"
            self.status_indicator.text_color = (1, 0.3, 0.3, 1)
            self.check_cameras_availability()
            return

        # Настройка параметров камеры
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.capture.set(cv2.CAP_PROP_FPS, 30)

        print(f"Камера {camera_index} успешно запущена")
        mode_text = "ЧИСТЫЕ" if not self.save_with_detection else "С ДЕТЕКЦИЕЙ"
        self.status_indicator.text = f"КАМЕРА АКТИВНА | СНИМКИ: {mode_text}"
        self.status_indicator.text_color = (0.3, 0.8, 0.3, 1)
        self.event = Clock.schedule_interval(self.update_frame, 1.0 / 15.0)

    def stop_video(self, *_):
        """Остановка видеопотока"""
        print("Остановка видеопотока")
        if self.event is not None:
            self.event.cancel()
            self.event = None
        if self.capture is not None:
            self.capture.release()
            self.capture = None

        mode_text = "ЧИСТЫЕ" if not self.save_with_detection else "С ДЕТЕКЦИЕЙ"
        self.status_indicator.text = f"КАМЕРА ОСТАНОВЛЕНА | СНИМКИ: {mode_text}"
        self.status_indicator.text_color = (1, 0.6, 0.1, 1)
        Clock.schedule_once(lambda dt: self.check_cameras_availability(), 2)

    def update_frame(self, *_):
        """Обновление кадра видео и обнаружение объектов"""
        if self.capture is None:
            return

        try:
            ret, frame = self.capture.read()
            if not ret:
                print("Ошибка чтения кадра")
                return

            self.last_frame = frame.copy()
            original_frame = frame.copy()

            # ИСПРАВЛЕНИЕ: Добавлена обработка ошибок в detect_objects
            detected_objects = detect_objects(frame)
            frame, detected_objects = draw_boxes(frame, detected_objects)

            # Подготовка кадра для отображения
            frame = cv2.flip(frame, 0)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.current_frame = original_frame

            # Обновление списка обнаруженных объектов
            self.update_objects_list(detected_objects)

            # Создание текстуры для отображения
            buf = frame.tobytes()
            texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='rgb')
            texture.blit_buffer(buf, colorfmt='rgb', bufferfmt='ubyte')
            self.video_display.texture = texture

        except Exception as e:
            print(f"Ошибка в update_frame: {e}")
            import traceback
            traceback.print_exc()

    def update_objects_list(self, detected_objects):
        """Обновление списка обнаруженных объектов с улучшенной фильтрацией"""
        current_time = datetime.now()
        
        # ОГРАНИЧЕНИЕ: обновляем не чаще чем раз в 1.5 секунды
        if self.last_log_time is not None:
            time_diff = (current_time - self.last_log_time).total_seconds()
            if time_diff < 1.5:
                # Не обновляем если время не прошло
                return
        
        self.last_log_time = current_time
        
        # ОСТАЛЬНОЙ КОД ФУНКЦИИ
        if not detected_objects:
            if not self.detected_objects_history:
                self.objects_list.clear_widgets()
                empty_item = DetectionListItem(text="Объекты не обнаружены")
                empty_item.text_color = (0.7, 0.7, 0.7, 1)
                self.objects_list.add_widget(empty_item)
                self.objects_count_label.text = "0 объектов"
            return

        current_time = datetime.now()

        # Создание уникальной сигнатуры для обнаруженных объектов
        object_counts = Counter(obj['class'] for obj in detected_objects)
        signature_parts = [f"{class_name}:{object_counts[class_name]}" for class_name in sorted(object_counts.keys())]
        current_signature = "|".join(signature_parts)

        if self.last_objects_signature == current_signature:
            return

        self.last_objects_signature = current_signature

        # Обновление интерфейса
        self.objects_list.clear_widgets()

        timestamp = current_time.strftime("%H:%M:%S")

        # Заголовок с общей статистикой
        total_objects = len(detected_objects)
        unique_types = len(object_counts)
        header_text = f"[ {timestamp} ]  ОБЪЕКТОВ: {total_objects}  ТИПОВ: {unique_types}"
        header_item = DetectionListItem(text=header_text)
        header_item.bg_color = (0.2, 0.3, 0.5, 1)
        header_item.text_color = (1, 1, 1, 1)
        self.objects_list.add_widget(header_item)

        self.objects_count_label.text = f"{total_objects} объектов"

        # Информация по каждому типу объектов
        for class_name, count in object_counts.items():
            confidences = [obj['confidence'] for obj in detected_objects if obj['class'] == class_name]
            avg_confidence = sum(confidences) / len(confidences)

            item_text = f"{class_name.upper()}: {count} шт. ({avg_confidence * 100:.0f}%)"

            list_item = DetectionListItem(text=item_text)

            # Цветовая индикация уверенности
            if avg_confidence > 0.8:
                list_item.text_color = (0, 0.7, 0, 1)  # зеленый - высокая уверенность
            elif avg_confidence > 0.5:
                list_item.text_color = (1, 0.6, 0, 1)  # оранжевый - средняя уверенность
            else:
                list_item.text_color = (1, 0.3, 0.3, 1)  # красный - низкая уверенность

            self.objects_list.add_widget(list_item)

        # Сохранение в историю
        self.detected_objects_history.append({
            'timestamp': timestamp,
            'objects': object_counts,
            'total': total_objects
        })

        # Ограничение размера истории
        if len(self.detected_objects_history) > 8:
            self.detected_objects_history = self.detected_objects_history[-8:]

    def save_snapshot(self, *_):
        """Сохранение скриншота в зависимости от выбранного типа"""
        if self.current_frame is not None:
            try:
                snapshot_frame = self.current_frame.copy()
                Clock.schedule_once(lambda dt: self._save_snapshot_async(snapshot_frame), 0.1)

                snapshot_type = "ЧИСТЫЙ" if not self.save_with_detection else "С ДЕТЕКЦИЕЙ"
                self.snapshot_button.text = "СОХРАНЯЕМ..."
                self.status_indicator.text = f"СОХРАНЕНИЕ {snapshot_type} СНИМКА..."
                self.status_indicator.text_color = (1, 0.8, 0.3, 1)

            except Exception as e:
                print(f"Ошибка при подготовке снимка: {e}")
                self.status_indicator.text = "ОШИБКА ПОДГОТОВКИ"
                self.status_indicator.text_color = (1, 0.3, 0.3, 1)
        else:
            print("Нет кадра для сохранения")
            self.status_indicator.text = "ОШИБКА: НЕТ КАДРА"
            self.status_indicator.text_color = (1, 0.3, 0.3, 1)

    def _save_snapshot_async(self, frame):
        """Асинхронное сохранение скриншота"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            if self.save_with_detection:
                # Сохранение снимка с детекцией
                filename = f"snapshot_detection_{timestamp}.jpg"
                detected_objects = detect_objects(frame)
                snapshot_with_boxes, detected_objects = draw_boxes(frame, detected_objects)
                self.add_info_panel(snapshot_with_boxes, detected_objects, timestamp)
                cv2.imwrite(filename, snapshot_with_boxes)
                print(f"Снимок с обнаруженными объектами сохранен как {filename}")
            else:
                # Сохранение чистого снимка
                filename = f"snapshot_clean_{timestamp}.jpg"
                self.add_basic_info_panel(frame, timestamp)
                cv2.imwrite(filename, frame)
                print(f"Чистый снимок сохранен как {filename}")

            Clock.schedule_once(lambda dt: self._on_snapshot_saved(filename), 0)

        except Exception as e:
            print(f"Ошибка при сохранении снимка: {e}")
            import traceback
            traceback.print_exc()
            Clock.schedule_once(lambda dt: self._on_snapshot_error(), 0)

    def _on_snapshot_saved(self, filename):
        """Обновление интерфейса после успешного сохранения"""
        snapshot_type = "ЧИСТЫЙ" if not self.save_with_detection else "С ДЕТЕКЦИЕЙ"
        self.snapshot_button.text = "СНИМОК"
        self.status_indicator.text = f"✅ {snapshot_type} СНИМОК: {filename}"
        self.status_indicator.text_color = (0.3, 0.8, 0.3, 1)

        Clock.schedule_once(lambda dt: self._restore_ui_state(), 2)

    def _on_snapshot_error(self):
        """Обновление интерфейса при ошибке сохранения"""
        self.snapshot_button.text = "СНИМОК"
        self.status_indicator.text = "ОШИБКА СОХРАНЕНИЯ"
        self.status_indicator.text_color = (1, 0.3, 0.3, 1)

        Clock.schedule_once(lambda dt: self._restore_ui_state(), 2)

    def _restore_ui_state(self):
        """Восстановление нормального состояния UI"""
        if self.event is not None:
            mode_text = "ЧИСТЫЕ" if not self.save_with_detection else "С ДЕТЕКЦИЕЙ"
            self.status_indicator.text = f"КАМЕРА АКТИВНА | СНИМКИ: {mode_text}"
            self.status_indicator.text_color = (0.3, 0.8, 0.3, 1)
        else:
            mode_text = "ЧИСТЫЕ" if not self.save_with_detection else "С ДЕТЕКЦИЕЙ"
            self.status_indicator.text = f"КАМЕРА ОСТАНОВЛЕНА | СНИМКИ: {mode_text}"
            self.status_indicator.text_color = (1, 0.6, 0.1, 1)

    def add_info_panel(self, frame, detected_objects, timestamp):
        """Добавление информационной панели к скриншоту с детекцией"""
        height, width = frame.shape[:2]

        panel_height = 100
        panel = np.zeros((panel_height, width, 3), dtype=np.uint8)
        panel.fill(40)

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        font_color = (255, 255, 255)
        thickness = 1

        title = f"YOLO Detection - {timestamp}"
        cv2.putText(panel, title, (10, 25), font, font_scale, font_color, thickness)

        if detected_objects:
            object_counts = Counter(obj['class'] for obj in detected_objects)
            stats_text = f"Objects: {len(detected_objects)} | Types: "
            types_list = []
            for obj_class, count in object_counts.items():
                types_list.append(f"{obj_class}({count})")
            stats_text += ", ".join(types_list[:4])
            if len(object_counts) > 4:
                stats_text += f" ... (+{len(object_counts) - 4})"
        else:
            stats_text = "No objects detected"

        cv2.putText(panel, stats_text, (10, 50), font, font_scale - 0.1, font_color, thickness)

        camera_name = "MAIN" if self.current_camera_index == 0 else "FRONT"
        details = f"Camera: {camera_name} | Model: YOLOv8"
        cv2.putText(panel, details, (10, 75), font, font_scale - 0.1, font_color, thickness)

        cv2.rectangle(panel, (0, 0), (width - 1, panel_height - 1), (100, 100, 100), 2)

        combined_frame = np.vstack([frame, panel])
        frame.resize(combined_frame.shape, refcheck=False)
        frame[:] = combined_frame

    def add_basic_info_panel(self, frame, timestamp):
        """Добавление базовой информационной панели к чистому снимку"""
        height, width = frame.shape[:2]

        panel_height = 60
        panel = np.zeros((panel_height, width, 3), dtype=np.uint8)
        panel.fill(40)

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        font_color = (255, 255, 255)
        thickness = 1

        title = f"Clean Snapshot - {timestamp}"
        cv2.putText(panel, title, (10, 25), font, font_scale, font_color, thickness)

        camera_name = "MAIN" if self.current_camera_index == 0 else "FRONT"
        details = f"Camera: {camera_name} | No object detection"
        cv2.putText(panel, details, (10, 45), font, font_scale - 0.1, font_color, thickness)

        cv2.rectangle(panel, (0, 0), (width - 1, panel_height - 1), (100, 100, 100), 2)

        combined_frame = np.vstack([frame, panel])
        frame.resize(combined_frame.shape, refcheck=False)
        frame[:] = combined_frame


class CameraApp(MDApp):
    """Главное приложение"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.theme_cls.primary_palette = "Blue"
        self.theme_cls.accent_palette = "Orange"
        self.theme_cls.theme_style = "Light"

    def build(self):
        """Создание главного окна приложения"""
        self.title = "YOLO Object Detection"
        return MainLayout()

    def on_pause(self):
        """Обработчик паузы приложения"""
        return True

    def on_resume(self):
        """Обработчик возобновления приложения"""
        pass


if __name__ == '__main__':
    CameraApp().run()
