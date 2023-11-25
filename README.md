# Atomic_fuel_prom_optimization

Решение основано на линейном программировании, и обучении с подкрелением, концептуально близкое к известному AlphaZero.

Строгая формализация позволяет обуславливатьтся на технологическую карту каждого взятого цеха, планировать тех.обслуживание станков без простоев производства, минимизировать ручные операции на каждой печи и продливать горизонт планирования на несколько дней вперед, учитывая так же приоритетность выполнения


## Воспроизведение
необходимо запустить файл `screencasted.py`, версия питона >= 3.10
при этом должны лежать json-ы в папке ./train/ относительно файла и места запуска

## Мат формализация, инсайты из бейзлана

##### Обозначения:
- $x$ - индекс машины
- $t$ - время
- $n$ - номер серии
- $T$ - температура
- $p>0$ - функция приоритета
- $l$ - лейбл кластера (для серийного производства)
- $o^k$ - флаг ковки
- $o^p$ - флаг проката

##### Константы: 
- $N_n = $ 7 - кол-во печей
- $N_k = $ 1 - кол-во ковок
- $N_p = $ 1 - кол-во проката
- $\delta t_{max} = $ 15м - макс время ожидания
- $\delta t_+ = $ 2ч - время нагрева 
- $\delta t_- = $ 1ч+15м - время охлаждения
- $\forall 0<x<N_n: t^n_{0,x} = -$ 1ч  $~~ t^n_{1,x} = $ 23ч  - время жизни всех печей
- $\forall 0<x<N_k: t^k_{0,x} = -$ 1ч  $~~ t^k_{1,x} = $ 23ч - время жизни всех ковок
- $\forall 0<x<N_p: t^p_{0,x} = $ 7ч   $~~~~~~ t^p_{1,x} = $ 23ч - время жизни всех прокатов

##### Ограничения:
- Температура и операция должна быть доступна на печи
- Смена температры = простой
- В серии могут быть разрывы не более $t_{max}$
- Кол-во ковки в момент времени ограничено и зависит от вреени
- Кол-во проката в момент времени ограничено и зависит от вреени
- Серии не перекрываются
- Серии целиком помещаются во время жизни печей

##### Упрощения:
- все машины одного типа имеют одинаковое время жинзи
- серии не переходят между печами

##### Соглашение:
- простой/заполененность заносится в приоритет ячейки расписания
- у пропусков и смены темературы приоритет ячейки нулевой

##### Формулировка задачи оптимизации:

Задача ставится на пространстве контреных печей и при произвольном списке серий.

Введем матрицу состояния, где каждая ячейка - печь в определенный момент времени (представление Эйлера), а я ячейке лежит информация о текующей серии, приоритете ячейки, кластризации серии, текущей температуре:
$$
    E(x, t) = (n, T, p, l, o^k, o^p), \quad 0 < x < N_n,\quad t^n_0 < t < max(t^n_1, t^k_1, t^p_1)
$$
в таком представлении все параметры $(n, T, p, l, o^k, o^p)$ переменные 

Перестановкой строк в столбце и добавлением новых строк можно сделать матрицу серия - время (представление Лагранжа):
$$
    L(n, t) = (x, T, p, l, o^k, o^p), \quad n\in\{n\},\quad t^n_0 < t < max(t^n_1, t^k_1, t^p_1)
$$
в таком представлении от времени зависят только операции $o^k, o^p$ и наличие пропусков в строке.

Если серия не назначена на печь, ее нет в $E$, а все ее ячейки в $L$ равны нулю.

Очевидное свойство: если функция завивисит от всего столбца, то для функции $L=\hat SE \approx E$, например $\sum_x E(x, t)_p = \sum_n L(n, t)_p$

Оптимизация:

\begin{split}
    max \quad\quad & \sum_x \sum_t E(x,t)_p + l*\sum_x m(E(x,\cdot)_l) \\

    subject~to \quad
    & \forall x, t ~~ \left( E(x,t+1)_T > E(x,t)_T \right) *\sum_{\tau = t - \delta t_+}^{\tau = t}E(x,\tau)_p = 0 \\
    & \forall x, t ~~ \left( E(x,t+1)_T < E(x,t)_T \right) *\sum_{\tau = t - \delta t_-}^{\tau = t}E(x,\tau)_p = 0 \\
    & \forall n, t ~~ \sum_{\tau = t}^{t + \delta t_{max}} L(n, \tau)_p > 0 \\
    & \forall t ~~ \sum_x E(x, t)_{o^k} <= N_k \\
    & \forall t ~~ \sum_x E(x, t)_{o^p} <= N_p \\
    & \forall x, t ~~ E(x,t)_p * m^T(t^n_0, t^n_1)(x,t) = 0\\
    & \forall x, t ~~ E(x,t)_{o^k} * m^T(t^k_0, t^k_1)(x,t) = 0\\
    & \forall x, t ~~ E(x,t)_{o^p} * m^T(t^p_0, t^p_1)(x,t) = 0\\
\end{split}

где $m$ - внутрянняя метрика класстеризации/кучности, $l$ - степень кучности, $m^T$ - маска времени жизни, равна нулю в пределах отрезка и равна единице за пределами.
При этом требование неперекрытия серий выполняется на станках заданными ограничениями и на печах в силу определения $E$
