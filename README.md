Python: module AUV\_control15  code { font-family: Consolas, Monaco, "Andale Mono", "Ubuntu Mono", monospace; -webkit-overflow-scrolling: touch; overflow-x: scroll; max-width: 100%; min-width: 100px; padding: 2px 5px 2px 5px; white-space: break-spaces; background-color: rgb(184, 197, 213); border-radius: 6px; } .man-title { color: #f80000; font-style: oblique; } .man-param { font-style: italic; } .man-return { font-style: italic; } .man-desc { -webkit-text-stroke: 0.5px; } .man-code-1 { font-family: Consolas, Monaco, "Andale Mono", "Ubuntu Mono", monospace; -webkit-overflow-scrolling: touch; overflow-x: scroll; max-width: 100%; min-width: 100px; padding: 2px 5px 2px 5px; white-space: break-spaces; background-color: rgb(184, 197, 213); border-radius: 6px; } .man-code-2 { font-family: Consolas, Monaco, "Andale Mono", "Ubuntu Mono", monospace; -webkit-overflow-scrolling: touch; overflow-x: scroll; max-width: 100%; min-width: 100px; padding: 0px; white-space: break-spaces; background-color: rgb(150, 150, 150); border-radius: 6px; }

   
 **AUV\_control15**

[index](.)  
[AUV\_control15.py](file:AUV_control15.py)

# coding: utf-8

   
**Modules**

      

 

[numpy](numpy.html)  

[os](os.html)  

[matplotlib.pyplot](matplotlib.pyplot.html)  

   
**Functions**

      

 

**abs\_value**(x: numpy.ndarray) -> float

Вычисляет абсолютную величину вектора x  
   
параметр `x`: массив np.ndarray значений  
возвращает: float длина вектора x

**ann\_train**(npdata, labels\_dict: dict, init\_weight\_list: list, init\_bias\_list: list, act\_func, mu: float = 0.05, epoch: int = 500) -> (<class 'list'>, <class 'list'>, <class 'list'>)

Реализация обучения нейронной сети методом обратного распространения ошибки с использованием градиентного спуска  
   
параметр `npdata`: массив данных для обучения ИНС  
параметр `labels_dict`: dict словарь меток для каждого слоя ИНС  
параметр `init_weight_list`: список np.ndarray массивов начальных весов ИНС  
параметр `init_bias_list`: список np.ndarray начальных смещений ИНС  
параметр `act_func`: func функция активации ИНС  
параметр `mu`: float скорость обучения ИНС  
параметр `epoch`: int количество эпох обучения ИНС (по умолчанию 500)  
возвращает: (list, list) обновленные веса начальных init\_weight\_list и смещения init\_bias\_list для ИНС

**cos**(x, /)

Return the cosine of x (measured in radians).

**filename\_fmt**(name: str, prefix\_path: str = './img', ext: str = 'png')

Форматирует имя файла для сохранения графика  
   
параметр `name`: str часть имени пути файла для сохранения  
параметр `prefix_path`: str директория пути сохранения (по умолчанию, ./img)  
параметр `ext`: str расширение файла (по умолчанию, png)  
возвращает: Форматированный путь

**generate\_f**(T: int, A: int, mexp: float = 0.5, stdiv: float = 0.2) -> list

Генерация случайных значений из нормального распределения с математическим ожиданием mexp  
и стандартным отклонением stdiv для входных данных  
   
параметр `T`: int длина данных  
параметр `A`: int ширина данных  
параметр `max_exp`: float математическое ожидание (по умолчанию, 0.5)  
параметр `stdiv`: float величина стандартного отклонения (по умолчанию, 0.2)  
возвращает: список np.array массивов случайных значений

**generate\_labels**(source\_layer\_arr: numpy.ndarray, source\_labels\_arr: numpy.ndarray, weight\_arr: numpy.ndarray) -> numpy.ndarray

Генерирование метки для обучения следующего слоя нейронной сети  
   
параметр `source_layer_arr`: np.ndarray вектор значений, полученный на выходе нейронной сети на определенном слое  
параметр `source_labels_arr`: np.ndarray вектор желаемых значений на данном слое  
параметр `weight_arr`: np.ndarray матрица весов, которая определяет связи между нейронами текущего слоя и предыдущего слоя  
возвращает: np.ndarray метки для обучения следующего слоя

**get\_n**(phi: int, psi: int) -> numpy.ndarray

Генерируют входные данные для нейронной сети на основе косинуса и синуса углов phi и psi  
для вычисления компонент вектора n в трехмерном пространстве  
   
параметр `phi`: int угол phi  
параметр `psi`: int угол psi  
возвращает: np.ndarray массив компонент вектора n в трехмерном пространстве  
   
Вектор n представляет собой единичный вектор, направление которого определяется углами phi и psi.  
Он может использоваться для различных целей, таких как определение направления движения объекта в пространстве  
или ориентации объекта относительно других объектов

**get\_x**(T: int) -> numpy.ndarray

Функция генерации входных данных для нейронной сети  
   
параметр `T`: int входное число  
возвращает:  массив np.ndarray размера Tx3 элементов целочисленного значения от 0 до T-1  
   
Функция принимает на вход число T и возвращает numpy массив размера Tx3 элементов  
целочисленного значения от 0 до T-1

**layer**(npdata: numpy.ndarray, weight\_arr: numpy.ndarray, bias\_arr: numpy.ndarray, act\_func) -> numpy.ndarray

Вычисление выходных значений слоя нейронной сети при заданных входных значениях npdata,  
матрице весов weight\_arr и векторе смещений bias\_arr с использованием функции активации act\_func  
   
параметр `npdata`: np.ndarray массив данных (входной сигнал), размерность которого равна (n, m), где: n - количество признаков, m - количество примеров  
параметр `weight_arr`: np.ndarray массив весов, размерность которого равна (n, k), где k - количество нейронов в слое  
параметр `bias_arr`: np.ndarray массив смещений, размерность которого равна (1, k)  
параметр `act_func`: func функция активации, применяемая к выходу каждого нейрона  
возвращает: np.ndarray массив, содержащий выходные значения нейронов в слое после применения функции активации к их суммарному входу. Размерность этого массива равна (m, k),  
                      где m - количество примеров, k - количество нейронов в слое.

**layer\_train**(npdata: numpy.ndarray, nplabels: numpy.ndarray, init\_weight\_arr: numpy.ndarray, init\_bias\_arr: numpy.ndarray, act\_func, mu: float, epoch: int = 100) -> (<class 'numpy.ndarray'>, <class 'numpy.ndarray'>, <class 'list'>)

Обучение одного слоя ИНС с использованием метода обратного распространения ошибки  
   
параметр `npdata`: numpy.ndarray массив входных данных  
параметр `nplabels`: numpy.ndarray массив меток  
параметр `init_weight_arr`: numpy.ndarray массив инициализированных весовых коэффициентов  
параметр `init_bias_arr`: numpy.ndarray массив инициализированных коэффициентов смещения  
параметр `act_func`: func функция активации  
параметр `mu`: float коэффициент скорости обучения  
параметр `epoch`: int количество эпох обучения  
возвращает: (np.ndarray, np.ndarray, list) обученные значения начальных весов init\_weight\_arr и смещений init\_bias\_arr.  
   
Функция принимает на вход матрицу входных данных npdata, матрицу ожидаемых выходных значений nplabels,  
начальные значения весов init\_weight\_arr и смещений init\_bias\_arr, функцию активации act\_func,  
коэффициент обучения mu и количество эпох epoch

**layers**(npdata: numpy.ndarray, weight\_list: list, bias\_list: list, act\_func) -> list

Функция вычисляет выходные значения для каждого слоя нейронной сети и возвращает список выходных  
значений для каждого слоя  
   
параметр `npdata`: np.ndarray массив, содержит входные данные для обработки  
параметр `weight_list`: список np.ndarray массивов, содержащих веса слоев нейронной сети  
параметр `bias_list`: список np.ndarray массивов, содержащих смещения слоев нейронной сети  
параметр `act_func`: func функция активации, которая будет применяться к выходу каждого слоя  
возвращает: список np.ndarray выходных значений для каждого слоя  
   
Функция принимает на вход матрицу входных данных npdata, список матриц весов weight\_list  
и список векторов смещений bias\_list для каждого слоя нейронной сети, а также функцию активации act\_func

**random**(...) method of [random.Random](random.html#Random) instance

[random](#-random)() -> x in the interval \[0, 1).

**sigmoid**(x: float) -> float

Вычисляет значение сигмоидной функции активации для каждого элемента входного массива x  
и возвращает массив со значениями сигмоидной функции активации  
   
параметр `x`: float входное значение  
возвращает: float результат сигмоидной функии

**sin**(x, /)

Return the sine of x (measured in radians).

**start**(i: int) -> float

Генерирует начальные значения для входных данных  
   
параметр `i`: int входящее число  
возвращает:  float 0 | 1 | random{-1, 1}  
   
Возвращает 0, 1 или случайное число (в интервале от -1 до 1) в зависимости от значения аргумента i:  
{-inf, 3}: 0  
{3, 6}: 1  
{6, 9}: 0  
{9, inf}: random{-1, 1}

**umpf**(step\_num: int, expected\_values\_list: list, current\_values\_arr: numpy.ndarray, gamma: int) -> numpy.ndarray

Вычисляет матрицу входных данных для каждого слоя нейронной сети  
   
параметр `step_num`: int номер текущего шага  
параметр `expected_values_list`: список np.ndarray массивов значений, представляющих собой целевые значения, которые должна достичь система  
параметр `current_values_arr`: массив np.ndarray значений, представляющих собой текущее состояние системы  
параметр `gamma`: int пороговое значение для функции управления при обучении ИНС  
возвращает: массив np.ndarray матрицы входных данных для каждого слоя нейронной сети  
   
В алгоритме происходит вычисление управляющего воздействия vul на текущем шаге времени,  
используя значения управляющего воздействия с предыдущего шага времени и состояния системы на текущем  
и предыдущем шагах времени. Если текущий шаг времени кратен Tp, то функция также вычисляет  
состояние системы на следующем шаге времени vpl1.  
Если текущий шаг времени кратен Tu, то функция вычисляет управляющее воздействие на следующем шаге времени vml

   
**Data**

      

 

**pi** = 3.141592653589793