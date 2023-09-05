# Autoencoder
## Результаты работы

1. Реализованы:

- get_arch() - построение архитектуры автоэнкодера
- save_model() - метод для сохранения модели на диск
- load_model() - метод длязагрузки модели с локального носителя
- build_encoder(), build_decoder() - вспомогательные методы для построения архитектур енкодера и декодера соответственно
- CustomCallback() - класс, реализующий визуализацию результатов обучения нейросети

2. Результаты экспериментов

Обучение проводилось на датасете cifar10. Размер изображений: 32 * 32.

В ходе экспериментов наилучшие результаты были получены при применении 2 уровней понижения размерности.

Дальнейшие результаты приведены при следующих условиях:

- кол-во уровней понижения - 2
- эпох - 6
- batch_size - 254

Результаты:

- использование слоя BatchNormalization. С использованием слоя loss = 0.0069, без него loss = 0.0083. Вывод: использование слоя улучшает работу нейросети.
- Применение MaxPooling2D: loss = 0.0083. Применение Conv2D(strides=(2, 2)): loss = 0.0059. Вывод: использование слоя Conv2D с шагом = 2 вместо использования MaxPooling при кодировании дает лучшие результаты.
- Применение Upsempling2D и Conv2d: loss = 0.0083. Применение Conv2dTranspose(strides=(2, 2)): loss = 0.0069. Вывод применение слоя Conv2dTranspose при декодировании дает более точные результаты чем UpSampling2D.
- Применение Conv2d(strides=(2, 2)) при кодировании, применение Conv2dTranspose(strides=(2, 2)) при декодировании: loss = 0.0042.


# Variational autoencoder
На основе автоэнкодера реализован вариационный автоэнкодер, генерирующий новые изображения на основе обучающего датасета изображений 
