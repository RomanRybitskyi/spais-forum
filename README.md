# SPAIS — Safety & Presence AI System

Система комп'ютерного зору реального часу на базі CPU для моніторингу безпеки.  
Два режими роботи:

- **Scenario 1** — виявлення людей, трекінг, визначення статі, наявності каски, аналіз присутності в зоні ROI
- **Scenario 2** — виявлення облич, визначення статі та емоційного стану

---

## Вимоги до системи

| Компонент | Мінімум |
|---|---|
| Python | 3.10+ |
| ОС | Windows 10/11, Ubuntu 20.04+ |
| Камера | USB або вбудована (index 0 або 1) |
| RAM | 4 GB |
| CPU | 4+ ядра (рекомендовано) |

> GPU **не потрібен** — всі моделі працюють на CPU через ONNX Runtime.

---

## 1. Клонування репозиторію

```bash
git clone https://github.com/RomanRybitskyi/spais-forum.git
cd spais-forum
```

---

## 2. Створення та активація віртуального середовища

**Windows (PowerShell):**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**Linux / macOS:**
```bash
python3 -m venv venv
source venv/bin/activate
```

> Якщо PowerShell блокує виконання скриптів, виконай один раз:
> ```powershell
> Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
> ```

---

## 3. Встановлення залежностей

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

> **Опціонально** — InsightFace (покращене визначення статі + вік):
> ```bash
> pip install insightface
> ```
> Після встановлення увімкни в `config.py`: `USE_INSIGHTFACE_GENDER = True`

> **Опціонально** — прискорення через Intel OpenVINO (тільки Intel CPU):
> ```bash
> pip install onnxruntime-openvino openvino
> ```
> Після встановлення увімкни в `config.py`: `USE_OPENVINO_EP = True`

---

## 4. Структура проекту

```
spais-forum/
├── config.py               # всі налаштування
├── main.py                 # запуск через OpenCV вікно (без браузера)
├── models/                 # ONNX-моделі (включені в репо)
│   ├── person_detector/
│   ├── gender_classifier/
│   ├── helmet_detector/
│   ├── face_detector/
│   └── emotion_classifier/
├── modules/                # модулі pipeline
│   ├── detector.py         # YOLOv8 детектор людей
│   ├── tracker.py          # ByteTrack трекер
│   ├── gender.py           # класифікатор статі
│   ├── helmet.py           # детектор каски
│   ├── face_detector.py    # YuNet детектор облич
│   ├── emotion.py          # класифікатор емоцій
│   ├── zone.py             # аналітика ROI-зони
│   └── renderer.py         # відображення
├── webapp/                 # веб-інтерфейс (FastAPI)
│   ├── server.py
│   ├── runner.py
│   └── static/             # HTML / JS / CSS
├── tools/                  # утиліти
│   └── define_roi.py       # інструмент для малювання зони
└── tests/                  # unit-тести
```

---

## 5. Налаштування камери

Відкрий `config.py` і встанови правильний індекс камери:

```python
CAMERA_INDEX = 0   # 0 — вбудована, 1 — перша зовнішня, тощо
```

Щоб дізнатися який індекс у твоєї камери, запусти:

```bash
python test.py
```

Скрипт перевірить індекси 0–4 і покаже які камери доступні.

---

## 6. Запуск — варіант A: веб-інтерфейс (рекомендовано)

```bash
python -m webapp.server
```

Або через uvicorn напряму:

```bash
uvicorn webapp.server:app --host 0.0.0.0 --port 8000
```

Після запуску відкрий браузер:

```
http://localhost:8000
```

### Управління через веб-інтерфейс

1. **Start** — запускає pipeline та відеопотік
2. **Stop** — зупиняє pipeline та звільняє камеру
3. **Scenario 1 / 2** — перемикання між режимами
4. **Draw Zone** — намалювати ROI-зону мишкою прямо на відео
   - Клікай щоб додати точки полігону
   - Подвійний клік або кнопка **Finish** — зберегти зону
   - **Esc** або кнопка **Cancel** — скасувати малювання
5. **Reset Zone** — повернути зону до стандартної
6. **Anonymise** — приховати мітки статі та ID
7. **Debug HUD** — показати/сховати панель з FPS та затримками

---

## 7. Запуск — варіант B: standalone OpenCV вікно

```bash
python main.py
```

| Клавіша | Дія |
|---|---|
| `q` | Вийти |
| `a` | Увімкнути/вимкнути анонімізацію |
| `d` | Увімкнути/вимкнути debug HUD |

---

## 8. Налаштування зони ROI вручну (інструмент)

Якщо хочеш попередньо задати зону без запуску основного pipeline:

```bash
python tools/define_roi.py
```

- Малюй зону кліками миші
- `Enter` або подвійний клік — зберегти
- `Esc` — скасувати
- Координати зони виводяться в консоль — скопіюй у `config.py` як `ROI_POLYGON`

---

## 9. Основні параметри `config.py`

| Параметр | За замовч. | Опис |
|---|---|---|
| `CAMERA_INDEX` | `0` | Індекс камери |
| `CAMERA_BACKEND` | `700` | `700`=DirectShow (Windows), `0`=авто |
| `CAPTURE_WIDTH` / `HEIGHT` | `1280×720` | Роздільна здатність відео |
| `PERSON_CONF_THRESHOLD` | `0.25` | Поріг впевненості детектора людей |
| `GENDER_BACKEND` | `"yolo"` | Бекенд: `"yolo"`, `"onnx"`, `"caffe"`, `"insightface"`, `"auto"` |
| `GENDER_CONF_THRESHOLD` | `0.65` | Мінімальна впевненість для виводу статі |
| `SHOW_AGE` | `True` | Показувати вік поруч зі статтю (тільки InsightFace) |
| `HELMET_CONF_THRESHOLD` | `0.2` | Поріг детектора каски |
| `CLASSIFIER_FRAME_STRIDE` | `2` | Оновлювати стать/каску кожен N-й кадр |
| `EMOTION_FRAME_STRIDE` | `6` | Оновлювати емоцію кожен N-й кадр |
| `ANONYMISE_OVERLAY` | `False` | Приховати мітки |
| `SHOW_DEBUG_HUD` | `True` | Показувати FPS і затримки |
| `USE_OPENVINO_EP` | `False` | Прискорення через Intel OpenVINO |
| `ROI_POLYGON` | прямокутник | Координати зони присутності |
