import numpy as np
import json
import math

TT = 10
sn = 1

Nn = 7 # печей
Nk = 1 # ковок
Np = 1 # прокаток
tw = 15//TT # макс время ожидание
ti = 120//TT # время нагрева
td = 75//TT # время охлаждения

tn0, tn1 = -60//TT, 23*60//TT # жизнь печи
tk0, tk1 = -60//TT, 23*60//TT # жизнь ковщика
tp0, tp1 = -60//TT, 23*60//TT # жизнь прокатчика

def t2i(t): return t - min(tn0, tk0, tp0)
def i2t(i): return i + min(tn0, tk0, tp0)
n2d = {'nagrev':0,'podogrev':1,'kovka':2,'prokat':3,'otzhig':4,'dt':5}

"""
    E[x,t] = [n, T, p, l, ok, op]
    где x - индекс печи из ovens
"""
Ix, In, IT, Ip, Il, Iok, Iop, Is = 0, 0, 1, 2, 3, 4, 5, 6


def check(E, ovens):
    assert len(E) == len(ovens)

    n0, n1 = int(E[:,:,In].min()), int(E[:,:,In].max())
    L = np.zeros((n1 - n0 + 1, E.shape[1], E.shape[2]))
    for x, oven in enumerate(ovens):
        for t in range(E.shape[1]):
            v = E[x, t].copy()
            n, v[Ix] = int(v[In]), x
            L[n - n0, t] = v

            if  (v[Iok] and 'kovka'  not in oven['operations']) or \
                (v[Iop] and 'prokat' not in oven['operations']) or \
                v[IT] not in oven['working_temps']:
                return False

    for line in E:
        dT = line[1:,IT] - line[:-1,IT]
        convi = np.convolve(line[:,Ip], np.ones((ti)), mode='full')[-dT.shape[0]:]
        convd = np.convolve(line[:,Ip], np.ones((td)), mode='full')[-dT.shape[0]:]
        if ((dT > 0)*convi > 0).any() or ((dT < 0)*convd > 0).any():
            return False

    for line in L:
        if line[:,Ip].max() == 0: continue
        filledline = line[np.argmax(line[:,Ip]): -np.argmax(line[::-1,Ip]), Ip]
        conv = np.convolve(filledline, np.ones((tw+1)))
        if np.any(conv == 0):
            return False

    if (E[:,:,Iok].sum(0) > Nk).any():
        return False

    if (E[:,:,Iop].sum(0) > Np).any():
        return False

    m = np.zeros(E.shape[:-1])
    m[:,t2i(tn0):t2i(tn1)] = 0
    if (E[:,:,Ip]*m).sum() > 0:
        return False

    m = np.zeros(E.shape[:-1])
    m[:,t2i(tk0):t2i(tk1)] = 0
    if (E[:,:,Iok]*m).sum() > 0:
        return False

    m = np.zeros(E.shape[:-1])
    m[:,t2i(tp0):t2i(tp1)] = 0
    if (E[:,:,Iop]*m).sum() > 0:
        return False

    return True


def priority_fn(n, N=None):
    return N-n if N else np.exp(-n)


class OvenGame():
    def __init__(self, ovens, series, kluster_power = 0):
        self.ovens = ovens
        self.series = series
        self.kluster_power = kluster_power
        self.over = False

        E = np.zeros((Nn, t2i(max(tn1, tk1, tp1)), 7))
        for x, oven in enumerate(ovens):
            E[x,:,IT] = oven['start_temp']
        self.history = [E]

    def valid_moves(self):
        # определения
        E = self.state()
        moves = []
        def r_place_fn(line, place, ops, v):
            op = ops[0]
            name, dt = op['name'], op['timing']
            o_k, o_p = name == 'kovka', name == 'prokat'

            # проверка что опция встает на место - не выходит за границу, не превышает ковки
            if o_k: # проверка что не вышли за пределы и влезаем в кавку
                if place + dt >= t2i(tk1) or \
                    np.any(E[:, place: place + dt, Iok].sum(0) >= Nk):
                    return []
            elif o_p: # проверка что не вышли за пределы и влезаем в прокат
                if place + dt >= t2i(tp1) or \
                    np.any(E[:, place: place + dt, Iop].sum(0) >= Np):
                    return []
            elif place + dt >= t2i(tn1): # проверка что не вышли за пределы
                return []

            w = v.copy()
            # нагрев = простой TODO
            if name == 'dt':
                v[Ip] = 0
            # высталяем на линию
            v[Iok], v[Iop], v[Is] = o_k, o_p, n2d[name]
            line[place:place + op['timing']] = v
            # если это последняя опреация, то просто ее возвращаем
            if len(ops) <= 1: return [line]

            # продолжаем заполнять линию разными вариантами
            new_lines = []
            for dtw in range(0, tw+1):
                new_lines += r_place_fn(line.copy(), place + dt + dtw, ops[1:], w)
            return new_lines

        # алгоритм
        for n, seria in enumerate(self.series):
            n += 1
            if n in E[:,:,0]: continue

            for x, oven in enumerate(self.ovens):
                required_temp, ops = seria['temperature'], seria['operations']
                # проверка температурых режимов
                if required_temp not in oven['working_temps']: continue
                # проверка операций
                f = True
                for op in ops:
                    if op['name'] in ['kovka', 'prokat', 'otzhig']:
                        f &= op['name'] in oven['operations']
                if not f: continue

                line = E[x]
                start = np.argmax(line[::-1,2]>0)
                # проверка что не кончилось место совсем, но хотя бы есть
                if start == 0:
                    if np.max(line[:,2]>0) != 0: continue
                    else: start = len(line)

                # параметры серии для заполнения
                length = sum([op['timing'] for op in ops])
                v = [n, required_temp, length + priority_fn(n, len(self.series)), required_temp, 0, 0, 0]

                # настрйока текущей температуры
                cur_temp = line[-1,IT]
                if required_temp > cur_temp:
                    # continue
                    ops = [{'name':'dt','timing':ti}] + ops
                    length += ti
                if required_temp < cur_temp:
                    # continue
                    ops = [{'name':'dt','timing':td}] + ops
                    length += td

                # v[Ip] /= length

                # рекурсивный поиск новых шагов
                lines = []
                for place_t in range(max(start - length, 60//TT)):
                    new_line = line.copy()
                    new_line[place_t:, IT] = required_temp
                    lines += r_place_fn(new_line, len(line) - start + place_t, ops, v.copy())
                # заполение выходного массива
                for line_ in lines:
                    move = E.copy()
                    move[x] = line_
                    moves.append(move)

                    # yield move #TODO

        if len(moves) == 0:
            self.over = True
        return moves

    def score(self):
        return self.score_(self.state())

    def score_(self, E):
        return E[:,:,Ip].sum() - self.kluster_power * E[:,:,Il].std(1).sum()

    def metric(self):
        E = self.state()
        nfilled = ( (E[:,:,Ip]!=0) * (E[:,:,Is]!=5) ).sum()
        numbers = np.unique(E[:,:,In].ravel())
        return nfilled / E.shape[0] / E.shape[1], numbers.shape[0]-1, numbers.mean()

    def make_move(self, move):
        self.history.append(move)

    def undo_move(self):
        self.history.pop()

    def state(self):
        return self.history[-1]


def correct_json(data):
    target = ['kovka', 'prokat']
    operation = {'name' : 'podogrev', 'timing' : 120}

    # вставка подогревов
    for s_ind in range(len(data['series'])):
        start = None
        length = 0
        for ind in range(len(data['series'][s_ind]['operations']) - 1):
            if data['series'][s_ind]['operations'][ind]['name'] in target and data['series'][s_ind]['operations'][ind+1]['name'] in target:
                if start is None: start = ind + 1
                length +=1
        while length != 0:
            data['series'][s_ind]['operations'].insert(start, operation.copy())
            start+=2
            length-=1

    # округление времени
    for s in data['series']:
        for op in s['operations']:
            op['timing'] = math.ceil(op['timing']/TT)


def get_game(N=100):
    i = np.random.randint(100)
    data = json.load(open(f'train/day-{i}.json', 'r'))
    correct_json(data)

    ovens = data['ovens']
    serias = data['series']

    ovens_ = np.random.choice(ovens, 7)
    series_ = np.random.choice(serias, N)

    return OvenGame(ovens_, series_)


def logshow(game):
    f, n, m = game.metric()
    print()
    print('Печи', '\tЗаполненность:', int(f*100), '   Кол-во серий', n, '   Средний номер серии:', int(m), '   Score:', game.score())
    print()
    E = game.state()

    for x in range(E.shape[0]):
        running_n = 0
        print(f'N{x} \t\033[47m', end='')
        for t in range(E.shape[1]):
            cell = E[x,t]
            print('', end='')
            if cell[In] > 0:
                match cell[Is]:
                    case 0: print('\033[48;2;245;194;66m', end='') # нагрев
                    case 1: print('\033[48;2;255;255;84m', end='') # подогрев
                    case 2: print('\033[48;2;159;206;99m', end='') # ковка
                    case 3: print('\033[48;2;104;55;155m', end='') # прокат
                    case 4: print('\033[48;2;228;126;51m', end='') # отжиг
                    case 5: print('\033[48;2;235;194;194m', end='') # изменение темпы

            if cell[In] != running_n and cell[Is] != 5 and cell[In] != 0:
                running_n = cell[In]
                print(chr(65 + int(running_n)%25) + ' '*(sn-1),'\033[47m', sep='', end='')
            else:
                print(' '*sn if cell[Ip] and cell[Is]!=5 else '-'*sn,'\033[47m', sep='', end='')

            if (t+1)%(60//TT)==0: print('|', end='')
        print('\033[0m')

    print()

    print('ковка \t', end='')
    for t, cell in enumerate(E[:,:,Iok].sum(0)):
        print('\033[48;2;159;206;99m'+' '*sn if cell else '\033[47m'+'-'*sn, end='')
        if (t+1)%(60//TT)==0: print('\033[47m|', end='')
    print('\033[0m')

    print()

    print('прокат \t', end='')
    for t, cell in enumerate(E[:,:,Iop].sum(0)):
        print('\033[48;2;104;55;155m'+' '*sn if cell else '\033[47m'+'-'*sn, end='')
        if (t+1)%(60//TT)==0: print('\033[47m|', end='')
    print('\033[0m')


if __name__=='__main__':
    for _ in range(3):
        game = get_game(500)

        while(True):
            moves = game.valid_moves()
            if len(moves) == 0: break
            # game.make_move(moves[np.argmin([ (m[Is]!=5).sum() for m in moves])])
            game.make_move(moves[np.argmax([ m[Ip].sum() for m in moves])])
            # game.make_move(moves[np.argmax(np.sum(moves, (1,2))[:,Ip])])
            # game.make_move(moves[np.argmax([game.score_(m) for m in moves])])
            # game.make_move(moves[0])

        logshow(game)
        print()
        print()
        print()