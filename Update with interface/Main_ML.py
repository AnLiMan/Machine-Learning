# ---------Библиотеки---------
import PIL.Image
import RawImageFromCamera
import tensorflow as tf
from tensorflow import keras
import time
import matplotlib.pyplot as plt
import os
import numpy as np
import math
import cv2
from PyQt6 import QtWidgets, uic, QtGui
import sys

# -------Постоянные--------
width = 1.9  # Ширина листопроката
velocity = 1.25  # Скорость листопроката м/с
cam_length = 4096  # Разрешение камеры по длине
cam_width = 3000  # Разрешение камеры по высоте
alpha = 72  # Угол обзора камеры
timing = time.time()

x1 = ['crazing', 'inclusion', 'patches', 'plitted_surface', 'rolled_in_scale', 'scratches']
x2 = ['crazing', 'inclusion', 'patches', 'plitted_surface', 'rolled_in_scale', 'scratches', "clear_surface"]

# -------Переменные--------
prediction_results = []  # Список результатов распознавания
text = []  # Буфер для хранения текста
time_for_1_operation = 0 #Время распознавания 1 тестового фрагмента
num_of_split_img = 0 # Количество фрагментов
captures_num = 0 # Количество кадров с камеры
OnWork = False #Флаг работы
time_frame = 0

#Количество дефектов
crazing = 0
inclusion = 0
patches = 0
plitted_surface = 0
rolled_in_scale = 0
scratches = 0
clean_surface = 0

vals = []  # Массив процентов
y1 = [] # Список количества дефектов
y2 = [] # Список количества дефектов и чистой поверхности

# -------------------------------------------------------
# ---------------Вспомогательные функции-----------------
# -------------------------------------------------------

# ---1. Вызов окна изображения с камеры------
def Cam_Window():
    RawImageFromCamera.Call_From_Other_Place()

#----2. Ссылки на исходники----
def Links():
    main_window.textBrowser.append("\n-----Ссылки-----")
    main_window.textBrowser.append("Ссылка на исходники на Git Hub")
    main_window.textBrowser.append("https://github.com/AnLiMan/Recognition-of-defects-on-the-metal-surface")
    main_window.textBrowser.append("\nСсылка на датасет")
    main_window.textBrowser.append("https://drive.google.com/file/d/1rDPzh5P2rKIaSV_PduEoMttUndGvpzrL/view?usp=share_link")
    main_window.textBrowser.append("\nСсылка на обученную модель")
    main_window.textBrowser.append("https://drive.google.com/file/d/1GqPEY834joSFFVflvwcuPNtI3gYBLwNl/view?usp=share_link")

#----3. Ссылка на фонд---
def Links2():
    main_window.textBrowser.append("\n----Спонсорская поддержка----")
    main_window.textBrowser.append("Прораммное обеспечение и исследования были проспонcированы Фондом Содействия Иновациям"
    " в рамках грантовой поддержки 'Умник', код 0081533, заявка  У-84718")
    main_window.textBrowser.append("\nСайт фонда")
    main_window.textBrowser.append("https://umnik.fasie.ru")

# ----3. Расчёт параметров камеры для настройки-----
def Cam_Parameters_Calculation():
    text = []
    b = width * cam_length / cam_width  # Длина участка полосы проката в м
    alpha_rad = alpha / 59.29577  # Из градусов в радианы
    edge_of_the_pyramid = b * math.sqrt(2 - 2 * math.cos(alpha_rad)) / (
            2 - 2 * math.cos(alpha_rad))  # Вычисление ребра пирамиды (стороны треугольника MON)
    cam_height = math.sqrt(edge_of_the_pyramid * edge_of_the_pyramid - (b / 4))  # Высота камеры над листом ОН
    length_for_pixel = b / cam_length  # Соотношение одного пикселя к длине полосы
    width_for_pixel = b / cam_width  # Соотношение одного пикселя к ширине полосы
    text.append("\n---Расчёт параметров камеры---")
    text.append('При длине полосы, попавшей в объектив = ' + str(round(b, 3)) + ' м')
    text.append('Высота камеры над листом = ' + str(round(cam_height, 3)) + ' м')
    text.append('Один пиксель соответствует длине листа в ' + str(round(length_for_pixel, 6)) + ' м')
    text.append(str('Один пиксель соответствует ширине листа в ' + str(round(width_for_pixel, 6)) + ' м'))

    # Посчитаем количество фрагментов для 1 кадра, для анализа
    number_of_fragments_in_length = int(cam_length / 200)
    text.append('Фрагментов по длине = ' + str(int(number_of_fragments_in_length)))
    number_of_fragments_in_width = int(cam_width / 200)
    text.append('Фрагментов по ширине = ' + str(number_of_fragments_in_width))
    text.append('Всего фрагментов = ' + str(number_of_fragments_in_length * number_of_fragments_in_width))
    time_for_frame_operation = number_of_fragments_in_length * number_of_fragments_in_width * time_for_1_operation
    text.append('Время на обработку одного кадра = ' + str(round(time_for_frame_operation, 3)) + ' c')

    # Расчёт промежутка времени через который будет делаться 1 кадр
    global time_frame
    time_frame = b / velocity  # Промежуток времени через который будет делаться снимок
    text.append('Промежуток времени через который будет делаться снимок = ' + str(round(time_frame, 4)) + ' c')
    text.append('Запас или недостаток (знак минус) по времени = ' + str(
        round((time_frame - time_for_frame_operation), 3)) + ' c')

    for i in range(len(text)):
        main_window.textBrowser.append(text[i])

# ------4. Создание папки, в которой будут сохраняться фрагменты----
def Create_Folder_For_Fragments():
    try:
        os.mkdir("Fragments")  # Создание пустого каталога (папки), в которую будут сохраняться фрагменты
        main_window.textBrowser.append("\nКаталог создан")
    except:
        main_window.textBrowser.append("\nКаталог уже создан")

# ------5. Удаление содержимого папки, в которой сохранялись фрагменты для временного анализа----
def Clean_Folder_For_Fragments():
    for the_file in os.listdir("Fragments"):
        file_path = os.path.join("Fragments", the_file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
            # elif os.path.isdir(file_path): shutil.rmtree(file_path)
        except Exception as e:
            print(e)

# ----6. Очистка поля textBrowser---
def CleanTextBrowser():
    main_window.textBrowser.clear()

#----7. Информация по работе----
def Info():
    main_window.textBrowser.append("\nПроцесс работы с програмой описан в файле Manual.pdf")
    main_window.textBrowser.append("Расшифровка индексов: clear surface = 0, crazing = 1, inclusion = 2, patches = 3, "
                                   "plitted surface = 4, rolled in scale = 5, scratches = 6")

#----8. Закрыть все окна----
def Close_All_Windows():
    sys.exit(app.exit())

# -----9. Разбиение снимка на фрагменты---------
def Split_Image():
    image = cv2.imread("Captures/Capture" + str(captures_num) + ".jpg")
    # image = cv2.imread("TestImage2.jpg") #Раскоментить для тестового прогона

    # Пересчёт количества фрагментов, на тот случай, если загружаемое изображение имеет иные размеры
    length = image.shape[0]  # Высота изображения
    width = image.shape[1]  # Ширина изображения
    # Посчитаем количество фрагментов для 1 кадра для анализа
    number_of_fragments_in_length_img = int(length / 200)
    number_of_fragments_in_width_img = int(width / 200)
    number_of_fragments_img = number_of_fragments_in_length_img * number_of_fragments_in_width_img
    global num_of_split_img
    num_of_split_img = number_of_fragments_img

    number = 0  # Вспомогательная переменная
    x_list = []  #

    # Дробление изображения
    for i in range(number_of_fragments_in_length_img):
        for k in range(number_of_fragments_in_width_img):
            number += 1
            x_list.append(number)
            cropped = image[(0 + 200 * i): (200 + 200 * i), (0 + 200 * k): (
                    200 + 200 * k)]  # Вырезаем изображения кусочками по 200х200 пикселей по всей площади
            name = "Cropped_" + str(number) + '.jpg'
            cv2.imwrite("Fragments/" + str(name), cropped)

    main_window.textBrowser.append("Всего фрагментов: " + str(number_of_fragments_img))

# ---10. Построение графиков дефектов----
def Build_Statistics():
    # Построим гистограмму
    fig, ax = plt.subplots()
    ax.bar(x1, y1)
    plt.title('Распределение дефектов')
    plt.xlabel('Дефект')
    plt.ylabel('Количество')
    ax.set_facecolor('white')
    fig.set_facecolor('white')
    fig.set_figwidth(10)  # Ширина графика
    fig.set_figheight(6)  # Высота графика
    plt.show()

    # Построим классический график
    plt.figure(figsize=(10, 6))
    plt.hlines(0, 0, 5)
    plt.xlim(0, 5)
    plt.grid()
    plt.title('Распределение дефектов')
    plt.xlabel('Дефект')
    plt.ylabel('Количество')
    plt.plot(x1, y1)
    plt.show()

    # Построим круговую диаграмму
    vals.append(y2[0])  # Волосные трещины
    vals.append(y2[1])  # Посторонние включения
    vals.append(y2[2])  # Пятна
    vals.append(y2[3])  # Рябизна
    vals.append(y2[4])  # Вдавленные окалины
    vals.append(y2[5])  # Царапины
    vals.append(y2[6])  # Чистая поверхность

    fig, ax = plt.subplots()
    ax.pie(vals, labels=x2)
    ax.legend(loc='upper right', bbox_to_anchor=(1.0, 1.0))
    ax.axis("equal")
    plt.show()

    main_window.textBrowser.append("Процент волосных трещин: " + str(round((y2[0] / len(prediction_results) * 100), 3)))
    main_window.textBrowser.append("Процент посторонних включений: " + str(round((y2[1] / len(prediction_results) * 100), 3)))
    main_window.textBrowser.append("Процент пятен: " + str(round((y2[2] / len(prediction_results) * 100), 3)))
    main_window.textBrowser.append("Процент рябизны: " + str(round((y2[3] / len(prediction_results)* 100), 3)))
    main_window.textBrowser.append("Процент вдавленных окалин: " + str(round((y2[4] / len(prediction_results) * 100), 3)))
    main_window.textBrowser.append("Процент царапин: " + str(round((y2[5] / len(prediction_results) * 100), 3)))
    main_window.textBrowser.append("Процент чистой поверхности: " + str(round((y2[6] / len(prediction_results ) * 100), 3)))

#----11. Сохранить отчёт-----
def Save_Report():
    date = str(main_window.calendarWidget.selectedDate())
    path = "Отчёт." + str(date[19:29])
    try:
        os.mkdir(path)
        main_window.textBrowser.append("\nПапка с отчётом создана")
    except Exception as e:
        main_window.textBrowser.append(str(e))
    f = open(str(path) + "/" + str(path) + ".rtf", "a+")
    try:
        f.seek(0)
        f.truncate()
        f.write("Отчёт от " + str(date[19:29]))
        f.write("\n")
        f.write(main_window.textBrowser.toPlainText())
        main_window.textBrowser.append("\nФайл отчёта создан")
    finally:
        f.close()

# ------12. Вынесение рекомендаций для прокатного оборудования----
def Recomendations_For_Equipment():
    max = y1[0]
    position = 0
    main_window.textBrowser.append("\n-----Вынесение рекомендаций на основе наиболее часто встречающихся дефектов-----")
    # Найдём индекс наибольшего значения в массиве
    for i in range(len(y1)):
        if y1[i] > max: max = y1[i]; position = i

    # Вынесение рекомендаций
    if (y1[0] > 1 or y1[3] > 1):
        main_window.textBrowser.append(
            '\nВнимание! Имеются волосные трещины и рябизна. Для повышения качества проката до обычного '
            'требуется провести шлифовку прокатных валов чистовой группы')
    elif (position == 1 or position == 4):
        main_window.textBrowser.append(
            '\nПреимущественный дефекты - посторонние включения и вдавленные окалины. Для улучшения качества '
            'поверности рекомендуется улучшить зачистку слябов, настроить работу горизонтального и вертикального '
            'окалиноломателя перед черновой группой клетей')
    elif (position == 2):
        main_window.textBrowser.append("Преимущественный дефект - пятна на поверности металла. Для улучшения качества поверности рекомендуется")
    elif (position == 5):
        main_window.textBrowser.append(
            '\nПреимущественный дефект - царапины на поверности металла. Для улучшения качества поверности рекомендуется '
            'удалить острые части направляющей арматуры или очистить поверхность валков чистовой группы')

# ------13. Присвоение категории качества по ГОСТ 5246-2016---
def Assigning_a_quality_category():
    if (y1[0] > 1 and y1[3] == 0):
        main_window.textBrowser.append('Категория качества по ГОСТ 5246-2016, табл. 10  - У')
    elif (y1[0] > 1 and y1[3] > 1):
        main_window.textBrowser.append('Категория качества по ГОСТ 5246-2016, табл. 10  - В')
    else:
        main_window.textBrowser.append('Категория качества по ГОСТ 5246-2016, табл. 10  - Обычная')

# ----14. Распознавание фрагментов нейросетью----
def Prediction():
    Create_Folder_For_Fragments()
    Split_Image()
    global OnWork
    OnWork = not OnWork
    main_window.textBrowser.append("\n-----Работа нейросети-----")

    for i in range(1, num_of_split_img + 1):
        name = "Cropped_" + str(i) + '.jpg'
        img = cv2.imread("Fragments/" + str(name))
        #img = PIL.Image.open("Fragments/" + str(name))
        array_img = np.array(img)
        norm_im = np.divide(array_img, 255)
        x_test = tf.expand_dims(norm_im, 0)
        prediction = CNN_Model.predict(x_test)
        pred_idx = np.argmax(prediction)
        prediction_results.append(pred_idx)
    main_window.textBrowser.append(str(prediction_results))
    Clean_Folder_For_Fragments()
    main_window.textBrowser.append("\n----Категория качества----")

    #Структуируем полученные данные
    global crazing, inclusion, patches, plitted_surface, rolled_in_scale, scratches, clean_surface

    # Посчитаем количество дефектов каждого вида
    for i in range(len(prediction_results)):
        if (prediction_results[i] == 0):
            clean_surface += 1
        elif prediction_results[i] == 1:
            crazing += 1
        elif prediction_results[i] == 2:
            inclusion += 1
        elif prediction_results[i] == 3:
            patches += 1
        elif prediction_results[i] == 4:
            plitted_surface += 1
        elif prediction_results[i] == 5:
            rolled_in_scale += 1
        elif prediction_results[i] == 6:
            scratches += 1

    # Добавим количество в массив
    y1.append(crazing)
    y1.append(inclusion)
    y1.append(patches)
    y1.append(plitted_surface)
    y1.append(rolled_in_scale)
    y1.append(scratches)

    y2.append(crazing)
    y2.append(inclusion)
    y2.append(patches)
    y2.append(plitted_surface)
    y2.append(rolled_in_scale)
    y2.append(scratches)
    y2.append(clean_surface)

    #Присвоим качегорию качества
    Assigning_a_quality_category()

#----15. Тест на скорость обработки и распознавания 1 фрагмента----
def SpeedRecognazionTest():
    main_window.textBrowser.append("\n-----Тест скорости обработки и распознавания 1 фрагмента-----")
    start_time = time.time()
    img = PIL.Image.open("patches_278.jpg")
    array_img = np.array(img)
    norm_im = np.divide(array_img, 255)
    x_test = tf.expand_dims(norm_im, 0)
    prediction = CNN_Model.predict(x_test)
    pred_idx = np.argmax(prediction)
    global time_for_1_operation
    time_for_1_operation = (time.time() - start_time)
    main_window.textBrowser.append("Время на обработку: " + str(round(time_for_1_operation, 3)) + " с")
    main_window.textBrowser.append("Индекс распознанного дефекта: " + str(pred_idx))

#------16. Создаём снимок с камеры
def CamCapture():
    global captures_num
    captures_num += 1
    cap = cv2.VideoCapture(0) # Подключаемся к камере. 0 — это индекс камеры, если их несколько то будет 0 или 1 и т.д.
    ret, img = cap.read()  # Читаем с устройства кадр, метод возвращает флаг ret (True , False) и img — саму картинку (массив numpy)
    cv2.imwrite("Captures/Capture" + str(captures_num) + ".jpg", img)
    cap.release()
    Prediction()

#-------17. Включение/отлючение работы компьютерного зрения---
def Work():
    global OnWork
    OnWork = not OnWork
    main_window.textBrowser.append("\nРабота компьютерного зрения = " + str(OnWork))

# --------------------------------------------
# -------------------Main---------------------
# --------------------------------------------
if __name__ == "__main__":
    CNN_Model = keras.models.load_model("Machine_learning_model_10.h5")
    app = QtWidgets.QApplication([])
    main_window = uic.loadUi("ProgramInterface.ui")
    main_window.setWindowTitle("Распознавание дефектов горячего проката")
    main_window.show()
    main_window.textBrowser.append("----Модель нейросети----")
    main_window.textBrowser.append("Machine_learning_model_10.h5")
    pixmap = QtGui.QPixmap('umnik_2.jpg')
    main_window.label_2.setPixmap(pixmap)

    # Отработка нажатия кнопок
    main_window.CameraCalculate.clicked.connect(Cam_Parameters_Calculation)
    main_window.ImageFromCamera.clicked.connect(Cam_Window)
    main_window.cleanBut.clicked.connect(CleanTextBrowser)
    main_window.SpeedTest.clicked.connect(SpeedRecognazionTest)
    main_window.Recomendations.clicked.connect(Recomendations_For_Equipment)
    main_window.SaveReport.clicked.connect(Save_Report)
    main_window.PlotsOut.clicked.connect(Build_Statistics)
    main_window.StartWork.clicked.connect(Prediction)

    #Отработка кнопок меню QAction
    main_window.Links.triggered.connect(Links)
    main_window.Acknowledgments.triggered.connect(Links2)
    main_window.AboutWork.triggered.connect(Info)
    main_window.CloseWindow.triggered.connect(Close_All_Windows)

    #Цикл работы
    if (OnWork and (time.time() - timing > time_frame)):
        print("Trigger")
        timing = time.time()

    sys.exit(app.exec())