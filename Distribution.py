import random
from tkinter import *
import time

colors = ["Black"]

#----------------------------------------------------------------------------------------------------------------------------------------------------------------#

open('cells.txt', 'w').close()
open('workers.txt', 'w').close()

def read_number_pairs(filename):
    result = []

    with open(filename, 'r') as file:
        for line in file:
            if not line.strip():
                continue

            parts = line.split()

            if len(parts) != 2:
                continue

            try:
                num1 = float(parts[0])
                num2 = float(parts[1])

                if num1.is_integer():
                    num1 = int(num1)
                if num2.is_integer():
                    num2 = int(num2)

                x = num1
                y = num2

                result += [[x, y]]

            except ValueError:
                continue

    ans = []

    for xr_yr in result:
        for xa_ya in ans:
            if (xa_ya == xr_yr):
                tk.mainloop()
                print("ERROR!!!")
                exit(0)
        ans += [xr_yr]

    return ans

#----------------------------------------------------------------------------------------------------------------------------------------------------------------#

global holst
holst = 1

def switch_canvas():
    global holst

    if canvas.winfo_viewable():
        holst = 2
        canvas.pack_forget()
        reference_canvas.pack()
        GO()
    else:
        holst = 1
        reference_canvas.pack_forget()
        canvas.pack()
        CLEAR()

def close_window():
    tk.destroy()

def get_button_width(button):
    return button.winfo_reqwidth()

def Make_1():
    tk.attributes('-fullscreen', False)

def Make_2():
    tk.attributes('-fullscreen', True)

#----------------------------------------------------------------------------------------------------------------------------------------------------------------#

tk = Tk()
tk.title('App')
tk.resizable(0, 0)
tk.wm_attributes('-topmost', 1)
tk.attributes('-fullscreen', True)

screen_width = tk.winfo_screenwidth()
screen_height = tk.winfo_screenheight()

size = int(min(screen_width, screen_height) * 6 / 7) // 20 * 20
l = size // 20

Gl = l
Gs = size

y1 = 0
x1 = Gl
while (y1 + Gl < screen_height):
    y1 += Gl
y1 -= Gl

y2 = Gl
x2 = 0
while (x2 + Gl < screen_width):
    x2 += Gl
y2 += Gl

n = (y1 - y2) // Gl
m = (x2 - x1) // Gl

global sn
sn = n

global sm
sm = m

global sx1
sx1 = x1

global sy1
sy1 = y1

global sx2
sx2 = x2

global sy2
sy2 = y2

global sl
sl = l

#----------------------------------------------------------------------------------------------------------------------------------------------------------------#

canvas = Canvas(tk, width=screen_width, height=screen_height, highlightthickness=0, background='old lace')
canvas.pack()

button_window = Button(canvas, text="Finish", command=close_window, bg='red')
button_window.place(x=screen_width-get_button_width(button_window), y=0)

b1 = Button(canvas, text="Window mode", command=Make_1, bg='red')
b1.place(x=screen_width - get_button_width(button_window) - l - get_button_width(b1), y=0)

b2 = Button(canvas, text="Fullscrean", command=Make_2, bg='red')
b2.place(x=screen_width - get_button_width(button_window) - l - get_button_width(b1) - l - get_button_width(b2), y=0)

b3 = Button(canvas, text="Find solution", command=switch_canvas, bg='red')
b3.place(x=screen_width - get_button_width(button_window) - l - get_button_width(b1) - l - get_button_width(b2) - l - get_button_width(b3), y=0)

for i in range (n + 1):
    canvas.create_line(x1, y1 - Gl * i, x2, y1 - Gl * i, fill="red", width=2)

for i in range (m + 1):
    canvas.create_line(x1 + Gl * i, y1, x1 + Gl * i, y2, fill="red", width=2)

canvas.create_line(x1, y1, x2, y1, fill="black", width=5)
canvas.create_line(x1, y2, x2, y2, fill="black", width=5)
canvas.create_line(x1, y2, x1, y1, fill="black", width=5)
canvas.create_line(x2, y1, x2, y2, fill="black", width=5)
canvas.create_line(x2 - Gl * 9, y1 - Gl * 7, x2 - Gl * 9, y2, fill="black", width=5)
canvas.create_line(x2 - Gl * 9, y1 - Gl * 7, x2 - 6 * Gl, y1 - Gl * 7, fill="black", width=5)
canvas.create_line(x2 - Gl * 4, y1 - Gl * 7, x2, y1 - Gl * 7, fill="black", width=5)

canvas.create_line(x2 - Gl * 4, y1 - Gl * 7, x2 - Gl * 6, y1 - Gl * 7, fill="deeppink", width=5)

arr = []

def create_area(x, y):
    global arr
    arr += [[x, y]]

    x1 = x - 2
    y1 = y + 1
    x2 = x - Gl + 1
    y2 = y + Gl - 2

    canvas.create_rectangle(x1, y1, x2, y2, fill="yellow", outline="yellow", stipple="gray25")

for i in range(n + 1):
    if (5 <= i and i <= 16):
        for j in range(m + 1):
            if (21 <= j and j <= 26):
                create_area(x1 + Gl * j, y1 - i * Gl)

mas = []

def create_cell(x, y):
    global mas
    mas += [[x, y]]

    x1 = x - 2
    y1 = y + 1
    x2 = x - Gl + 1
    y2 = y + Gl - 2

    canvas.create_rectangle(x1, y1, x2, y2, fill="brown", outline="orange")

for i in range(n + 1):
    if (i == 2 or i == 3 or i == 5 or i == 6 or i == 8 or i == 9 or i == 12 or i == 13 or i == 15 or i == 16 or i == 18 or i == 19):
        for j in range(m + 1):
            if (2 <= j and j <= 5 or 7 <= j and j <= 9 or 12 <= j and j <= 14 or 16 <= j and j <= 19):
                create_cell(x1 + Gl * i, y1 - j * Gl)

for i in range(n + 1):
    if (18 <= i and i <= 19):
        for j in range(m + 1):
            if (21 <= j and j <= 26):
                create_cell(x1 + Gl * j, y1 - i * Gl)

for i in range(n + 1):
    if (2 <= i and i <= 3):
        for j in range(m + 1):
            if (28 <= j and j <= 32):
                create_cell(x1 + Gl * j, y1 - i * Gl)

for i in range(n + 1):
    if (2 <= i and i <= 6):
        for j in range(m + 1):
            if (34 <= j and j <= 35):
                create_cell(x1 + Gl * j, y1 - i * Gl)

for i in range(n + 1):
    if (9 <= i and i <= 11 or 13 <= i and i <= 15 or 17 <= i and i <= 19):
        for j in range(m + 1):
            if (29 <= j and j <= 30):
                create_cell(x1 + Gl * j, y1 - i * Gl)

for i in range(n + 1):
    if (i == 9 or i == 10 or i == 12 or i == 13 or i == 15 or i == 16 or i == 18 or i == 19):
        for j in range(m + 1):
            if (33 <= j and j <= 35):
                create_cell(x1 + Gl * j, y1 - i * Gl)

#----------------------------------------------------------------------------------------------------------------------------------------------------------------#

global cnt_workers
cnt_workers = 0
workers = {}

global cnt_cells
cnt_cells = 0
cells = {}

WORKER_COLOR = {}

def create_worker(x, y):
    global cnt_workers, WORKER_COLOR

    fl = read_number_pairs('workers.txt')
    fl += [[x, y]]

    open('workers.txt', 'w').close()

    F = open('workers.txt', 'a')

    for x_y in fl:
        x_ = x_y[0]
        y_ = x_y[1]
        F.write(str(x_) + " " + str(y_) + "\n")

    F.close()

    x1 = x - 2
    y1 = y + 1
    x2 = x - Gl + 1
    y2 = y + Gl - 2

    sss = colors[random.randint(0, 1000) % len(colors)]

    workers[cnt_workers] = canvas.create_oval(x1, y1, x2, y2, fill=sss, outline="orange")
    WORKER_COLOR[cnt_workers] = sss
    cnt_workers += 1

def create_cell(x, y):
    global cnt_cells

    fl = read_number_pairs('cells.txt')
    fl += [[x, y]]

    open('cells.txt', 'w').close()

    F = open('cells.txt', 'a')

    for x_y in fl:
        x_ = x_y[0]
        y_ = x_y[1]
        F.write(str(x_) + " " + str(y_) + "\n")

    F.close()

    x1 = x - 2
    y1 = y + 1
    x2 = x - Gl + 1
    y2 = y + Gl - 2

    cells[cnt_cells] = canvas.create_rectangle(x1, y1, x2, y2, fill="yellow", outline="orange")
    cnt_cells += 1

def delete_worker(x, y):
    global cnt_workers

    fl = read_number_pairs('workers.txt')

    open('workers.txt', 'w').close()

    cur = 0
    for x_y in fl:
        x_ = x_y[0]
        y_ = x_y[1]
        if (x == x_ and y == y_):
            break
        else:
            cur += 1

    canvas.delete(workers[cur])

    F = open('workers.txt', 'a')

    for x_y in fl:
        x_ = x_y[0]
        y_ = x_y[1]
        if (x == x_ and y == y_):
            pass
        else:
            F.write(str(x_) + " " + str(y_) + "\n")

    F.close()

    for i in range(cur, cnt_workers - 1):
        workers[i] = workers[i + 1]

    cnt_workers -= 1

def delete_cell(x, y):
    global cnt_cells

    fl = read_number_pairs('cells.txt')

    open('cells.txt', 'w').close()

    cur = 0
    for x_y in fl:
        x_ = x_y[0]
        y_ = x_y[1]
        if (x == x_ and y == y_):
            break
        else:
            cur += 1

    canvas.delete(cells[cur])

    F = open('cells.txt', 'a')

    for x_y in fl:
        x_ = x_y[0]
        y_ = x_y[1]
        if (x == x_ and y == y_):
            pass
        else:
            F.write(str(x_) + " " + str(y_) + "\n")

    F.close()

    for i in range(cur, cnt_cells - 1):
        cells[i] = cells[i + 1]

    cnt_cells -= 1

def W(x, y):
    fl = read_number_pairs('workers.txt')
    F = True

    for x_y in fl:
        if (x_y == [x, y]):
            delete_worker(x, y)
            F = False

        if (F == False):
            break

    if (F == True):
        create_worker(x, y)
    else:
        pass

def N(x, y):
    fl = read_number_pairs('cells.txt')
    F = True

    for x_y in fl:
        if (x_y == [x, y]):
            delete_cell(x, y)
            F = False

        if (F == False):
            break

    if (F == True):
        create_cell(x, y)
    else:
        pass

# ----------------------------------------------------------------------------------------------------------------------------------------------------------------#

def clicked(event):
    global arr, mas

    click_x = event.x
    click_y = event.y

    for i in range(len(arr)):
        x1 = arr[i][0]
        y1 = arr[i][1]
        x2 = x1 - Gl
        y2 = y1 + Gl

        F = True

        if (x2 <= click_x and click_x <= x1):
            if (y1 <= click_y and click_y <= y2):
                W(x1, y1)
                F = False

        if (F == False):
            break

    for i in range(len(mas)):
        x1 = mas[i][0]
        y1 = mas[i][1]
        x2 = x1 - Gl
        y2 = y1 + Gl

        F = True

        if (x2 <= click_x and click_x <= x1):
            if (y1 <= click_y and click_y <= y2):
                N(x1, y1)
                F = False

        if (F == False):
            break

canvas.bind("<Button-1>", clicked)

#----------------------------------------------------------------------------------------------------------------------------------------------------------------#

reference_canvas = Canvas(tk, width=screen_width, height=screen_height, highlightthickness=0, background='old lace')
reference_canvas.pack()

button_window_2 = Button(reference_canvas, text="Finish", command=close_window, bg='red')
button_window_2.place(x=screen_width-get_button_width(button_window_2), y=0)

b1 = Button(reference_canvas, text="Window mode", command=Make_1, bg='red')
b1.place(x=screen_width - get_button_width(button_window_2) - l - get_button_width(b1), y=0)

b2 = Button(reference_canvas, text="Fullscrean", command=Make_2, bg='red')
b2.place(x=screen_width - get_button_width(button_window_2) - l - get_button_width(b1) - l - get_button_width(b2), y=0)

Button_switch_2 = Button(reference_canvas, text="To the main page", command=switch_canvas)
Button_switch_2.place(x=screen_width - get_button_width(button_window_2) - l - get_button_width(b1) - l - get_button_width(b2) - l - get_button_width(Button_switch_2), y=0)

for i in range (n + 1):
    reference_canvas.create_line(x1, y1 - Gl * i, x2, y1 - Gl * i, fill="red", width=2)

for i in range (m + 1):
    reference_canvas.create_line(x1 + Gl * i, y1, x1 + Gl * i, y2, fill="red", width=2)

def create_area(x, y):
    x1 = x - 2
    y1 = y + 1
    x2 = x - Gl + 1
    y2 = y + Gl - 2

    reference_canvas.create_rectangle(x1, y1, x2, y2, fill="yellow", outline="yellow", stipple="gray25")

for i in range(n + 1):
    if (5 <= i and i <= 16):
        for j in range(m + 1):
            if (21 <= j and j <= 26):
                create_area(x1 + Gl * j, y1 - i * Gl)

def create_cell_reference_canvas(x, y):
    x1 = x - 2
    y1 = y + 1
    x2 = x - Gl + 1
    y2 = y + Gl - 2

    reference_canvas.create_rectangle(x1, y1, x2, y2, fill="brown", outline="orange")

for i in range(len(mas)):
    x = mas[i][0]
    y = mas[i][1]
    create_cell_reference_canvas(x, y)

reference_canvas.create_line(x1, y1, x2, y1, fill="black", width=5)
reference_canvas.create_line(x1, y2, x2, y2, fill="black", width=5)
reference_canvas.create_line(x1, y2, x1, y1, fill="black", width=5)
reference_canvas.create_line(x2, y1, x2, y2, fill="black", width=5)
reference_canvas.create_line(x2 - Gl * 9, y1 - Gl * 7, x2 - Gl * 9, y2, fill="black", width=5)
reference_canvas.create_line(x2 - Gl * 9, y1 - Gl * 7, x2 - 6 * Gl, y1 - Gl * 7, fill="black", width=5)
reference_canvas.create_line(x2 - Gl * 4, y1 - Gl * 7, x2, y1 - Gl * 7, fill="black", width=5)

reference_canvas.create_line(x2 - Gl * 4, y1 - Gl * 7, x2 - Gl * 6, y1 - Gl * 7, fill="deeppink", width=5)

#----------------------------------------------------------------------------------------------------------------------------------------------------------------#

global solve_cells_cnt
solve_cells_cnt = 0
solve_cells = {}

global solve_workers_cnt
solve_workers_cnt = 0
solve_workers = {}

global done_cells_cnt
done_cells_cnt = 0
done_cells = {}

NEED = []

def create_need_cell_reference_canvas(x, y):
    global done_cells_cnt

    x1 = x - 2
    y1 = y + 1
    x2 = x - Gl + 1
    y2 = y + Gl - 2

    rect_id = reference_canvas.create_rectangle(x1, y1, x2, y2, fill="yellow", outline="orange")
    done_cells[(x, y)] = rect_id
    done_cells_cnt += 1

def create_worker_reference_canvas(x, y):
    global solve_workers_cnt, WORKER_COLOR

    x1 = x - 2
    y1 = y + 1
    x2 = x - Gl + 1
    y2 = y + Gl - 2

    solve_workers[solve_workers_cnt] = reference_canvas.create_oval(x1, y1, x2, y2, fill=WORKER_COLOR[solve_workers_cnt], outline="orange")
    solve_workers_cnt += 1

def Draw_Done(x, y):
    if (x, y) in done_cells:
        rect_id = done_cells[(x, y)]
        reference_canvas.itemconfig(rect_id, fill="darkseagreen3")

def Draw_Done_back(x, y):
    if (x, y) in done_cells:
        rect_id = done_cells[(x, y)]
        reference_canvas.itemconfig(rect_id, fill="brown")

def Done(ij):
    global NEED

    i = ij[0]
    j = ij[1]
    ans = []
    for p in NEED:
        xx = p[0]
        yy = p[1]
        if (xx == i and yy == j):
            pass
        else:
            ans += [(xx, yy)]
    if (NEED == ans):
        return

    NEED = ans

    global convenient_indexes_to, convenient_indexes_from
    p = convenient_indexes_from[(i, j)]
    xx = p[0]
    yy = p[1]
    Draw_Done(xx, yy)

# ----------------------------------------------------------------------------------------------------------------------------------------------------------------#

def GO():
    get_cells = read_number_pairs('cells.txt')
    get_workers = read_number_pairs('workers.txt')

    for i in range(len(get_cells)):
        x = get_cells[i][0]
        y = get_cells[i][1]
        create_need_cell_reference_canvas(x, y)

    for i in range(len(get_workers)):
        x = get_workers[i][0]
        y = get_workers[i][1]
        create_worker_reference_canvas(x, y)

    Transform(get_cells, get_workers)

def CLEAR():
    global solve_cells_cnt
    global solve_workers_cnt
    global done_cells_cnt

    global solve_cells
    global solve_workers
    global done_cells

    for (x, y) in done_cells:
        Draw_Done_back(x, y)

    for i in range(solve_cells_cnt - 1, -1, -1):
        reference_canvas.delete(solve_cells[i])

    solve_cells_cnt = 0

    for i in range(solve_workers_cnt - 1, -1, -1):
        reference_canvas.delete(solve_workers[i])

    solve_workers_cnt = 0

    done_cells.clear()
    done_cells_cnt = 0

# ----------------------------------------------------------------------------------------------------------------------------------------------------------------#

convenient_indexes_to = {}
convenient_indexes_from = {}

def Find(mas, q):
    for p in mas:
        if (p == q):
            return True
    return False

def Transform(my_cells, my_workers):
    global sn, sm, sx1, sy1, sx2, sy2, sl
    global convenient_indexes_to, convenient_indexes_from

    for i in range(1, sn + 1):
        for j in range(1, sm + 1):
            y = sy1 - Gl * i
            x = sx1 + Gl * j

            convenient_indexes_to[(x, y)] = (sn - i, j - 1)
            convenient_indexes_from[(sn - i, j - 1)] = (x, y)

    comf_cells = []
    for cell in my_cells:
        xx = cell[0]
        yy = cell[1]
        p = convenient_indexes_to[(xx, yy)]
        xx_ = p[0]
        yy_ = p[1]
        comf_cells += [[xx_, yy_]]

    comf_workers = []
    for worker in my_workers:
        xx = worker[0]
        yy = worker[1]
        p = convenient_indexes_to[(xx, yy)]
        xx_ = p[0]
        yy_ = p[1]
        comf_workers += [[xx_, yy_]]

    global mas

    comf_mas = []
    for elem in mas:
        xx = elem[0]
        yy = elem[1]
        p = convenient_indexes_to[(xx, yy)]
        xx_ = p[0]
        yy_ = p[1]
        comf_mas += [[xx_, yy_]]

    graph = {}

    for i in range(sn):
        for j in range(sm):
            graph[(i, j)] = []

            if Find(comf_mas, [i, j]):
                if i - 1 >= 0 and not Find(comf_mas, [i - 1, j]):
                    graph[(i, j)].append((i - 1, j))
                if j - 1 >= 0 and not Find(comf_mas, [i, j - 1]):
                    graph[(i, j)].append((i, j - 1))
                if i + 1 < sn and not Find(comf_mas, [i + 1, j]):
                    graph[(i, j)].append((i + 1, j))
                if j + 1 < sm and not Find(comf_mas, [i, j + 1]):
                    graph[(i, j)].append((i, j + 1))
            else:
                if i - 1 >= 0 and (Find(comf_mas, [i - 1, j]) == False or Find(comf_cells, [i - 1, j]) == True):
                    if (i == 13 and (27 <= j and j <= 29 or 32 <= j and j <= 35)):
                        pass
                    else:
                        graph[(i, j)].append((i - 1, j))
                if j - 1 >= 0 and (Find(comf_mas, [i, j - 1]) == False or Find(comf_cells, [i, j - 1]) == True):
                    if (j == 27 and 0 <= i and i <= 12):
                        pass
                    else:
                        graph[(i, j)].append((i, j - 1))
                if i + 1 < sn  and (Find(comf_mas, [i + 1, j]) == False or Find(comf_cells, [i + 1, j]) == True):
                    if (i == 12 and (27 <= j and j <= 29 or 32 <= j and j <= 35)):
                        pass
                    else:
                        graph[(i, j)].append((i + 1, j))
                if j + 1 < sm and (Find(comf_mas, [i, j + 1]) == False or Find(comf_cells, [i, j + 1]) == True):
                    if (j == 26 and 0 <= i and i <= 12):
                        pass
                    else:
                        graph[(i, j)].append((i, j + 1))

    purpose = []
    for p in comf_cells:
        xx = p[0]
        yy = p[1]
        purpose += [(xx, yy)]

    global NEED
    NEED = purpose

    cheliks = []
    for p in comf_workers:
        xx = p[0]
        yy = p[1]
        cheliks += [(xx, yy)]

    Solve(graph, cheliks, purpose)

# ----------------------------------------------------------------------------------------------------------------------------------------------------------------#

from collections import deque
import math

def bfs(graph, start):
    dist = {start: 0}
    queue = deque([start])

    while queue:
        node = queue.popleft()
        for neighbor in graph.get(node, []):
            if neighbor not in dist:
                dist[neighbor] = dist[node] + 1
                queue.append(neighbor)
    return dist

def solve(graph, cheliks, purpose, t):
    all_points = cheliks + list(purpose)

    dist_matrix = {}
    point_index = {}
    for i, p in enumerate(all_points):
        point_index[p] = i
        dists = bfs(graph, p)
        for q in all_points:
            dist_matrix[(p, q)] = dists.get(q, float('inf'))

    k = len(cheliks)
    clusters = [[] for _ in range(k)]
    routes = [[start] for start in cheliks]
    worker_times = [0] * k

    target_scores = []
    for target in purpose:
        min_dist = min(dist_matrix[(start, target)] for start in cheliks)
        target_scores.append((min_dist, target))
    target_scores.sort(reverse=True)

    for _, target in target_scores:
        best_worker = None
        best_time = float('inf')
        best_route = None

        for i in range(k):
            current_route = routes[i]
            min_increase = float('inf')

            for pos in range(len(current_route)):
                if pos == 0:
                    cost = dist_matrix[(current_route[0], target)]
                else:
                    prev = current_route[pos - 1]
                    next_node = current_route[pos]
                    cost = (dist_matrix[(prev, target)] +
                            dist_matrix[(target, next_node)] -
                            dist_matrix[(prev, next_node)])

                if cost < min_increase:
                    min_increase = cost

            new_time = worker_times[i] + min_increase + t

            if new_time < best_time:
                best_time = new_time
                best_worker = i
                best_route = current_route.copy()
                best_route.append(target)

        clusters[best_worker].append(target)
        routes[best_worker] = best_route
        worker_times[best_worker] = best_time

    def route_distance(route):
        total = 0
        for i in range(1, len(route)):
            total += dist_matrix[(route[i - 1], route[i])]
        return total

    optimized_routes = []
    final_times = []

    for i in range(k):
        if len(routes[i]) <= 2:
            optimized = routes[i]
        else:
            optimized = two_opt_optimize(routes[i], dist_matrix)

        optimized_routes.append(optimized)
        final_time = route_distance(optimized) + len(clusters[i]) * t
        final_times.append(final_time)

    initial_temp = 100.0
    cooling_rate = 0.995
    num_iterations = 1000
    sa_routes = simulated_annealing(
        optimized_routes,
        dist_matrix,
        t,
        initial_temp,
        cooling_rate,
        num_iterations
    )

    final_times = []
    for route in sa_routes:
        dist = calculate_route_cost(route, dist_matrix)
        final_times.append(dist + (len(route) - 1) * t)

    max_time = max(final_times) if final_times else 0
    return max_time, sa_routes

def two_opt_optimize(route, dist_matrix):
    best = route
    best_cost = calculate_route_cost(route, dist_matrix)
    improved = True

    while improved:
        improved = False
        for i in range(1, len(route) - 1):
            for j in range(i + 1, len(route)):
                new_route = route[:i] + route[i:j + 1][::-1] + route[j + 1:]
                new_cost = calculate_route_cost(new_route, dist_matrix)

                if new_cost < best_cost:
                    best = new_route
                    best_cost = new_cost
                    improved = True
        route = best
    return best

def calculate_route_cost(route, dist_matrix):
    cost = 0
    for i in range(1, len(route)):
        cost += dist_matrix[(route[i - 1], route[i])]
    return cost

def simulated_annealing(initial_routes, dist_matrix, t, initial_temp, cooling_rate, num_iterations):
    current_routes = [route.copy() for route in initial_routes]
    best_routes = [route.copy() for route in initial_routes]
    current_cost = compute_makespan(current_routes, dist_matrix, t)
    best_cost = current_cost

    temp = initial_temp

    for i in range(num_iterations):
        new_routes = generate_neighbor(current_routes, dist_matrix)
        new_cost = compute_makespan(new_routes, dist_matrix, t)

        delta = new_cost - current_cost

        if delta < 0 or random.random() < math.exp(-delta / temp):
            current_routes = new_routes
            current_cost = new_cost
            if new_cost < best_cost:
                best_routes = new_routes
                best_cost = new_cost

        temp *= cooling_rate

    return best_routes

def compute_makespan(routes, dist_matrix, t):
    max_time = 0
    for route in routes:
        if len(route) == 0:
            continue
        dist = 0
        for i in range(1, len(route)):
            dist += dist_matrix[(route[i - 1], route[i])]
        total_time = dist + (len(route) - 1) * t
        if total_time > max_time:
            max_time = total_time
    return max_time

def generate_neighbor(routes, dist_matrix):
    new_routes = [route.copy() for route in routes]
    k = len(new_routes)

    valid_routes = [i for i in range(k) if len(new_routes[i]) > 1]

    if not valid_routes:
        return new_routes

    if random.random() < 0.5:
        donor_idx = random.choice(valid_routes)

        recipient_candidates = [i for i in range(k) if i != donor_idx]
        if not recipient_candidates:
            return new_routes
        recipient_idx = random.choice(recipient_candidates)

        point_idx = random.randint(1, len(new_routes[donor_idx]) - 1)
        point = new_routes[donor_idx][point_idx]

        del new_routes[donor_idx][point_idx]

        if len(new_routes[recipient_idx]) == 1:
            new_routes[recipient_idx].append(point)
        else:
            best_pos = find_best_insertion_pos(new_routes[recipient_idx], point, dist_matrix)
            new_routes[recipient_idx].insert(best_pos, point)

    else:
        if len(valid_routes) < 2:
            return new_routes

        idx1, idx2 = random.sample(valid_routes, 2)

        pos1 = random.randint(1, len(new_routes[idx1]) - 1)
        pos2 = random.randint(1, len(new_routes[idx2]) - 1)
        point1 = new_routes[idx1][pos1]
        point2 = new_routes[idx2][pos2]

        new_routes[idx1][pos1] = point2
        new_routes[idx2][pos2] = point1

    return new_routes

def find_best_insertion_pos(route, point, dist_matrix):
    best_pos = 1
    min_increase = float('inf')

    for pos in range(1, len(route) + 1):
        if pos == 1:
            increase = (dist_matrix[(route[0], point)] +
                        dist_matrix[(point, route[1])] -
                        dist_matrix[(route[0], route[1])])
        elif pos == len(route):
            increase = dist_matrix[(route[-1], point)]
        else:
            increase = (dist_matrix[(route[pos - 1], point)] +
                        dist_matrix[(point, route[pos])] -
                        dist_matrix[(route[pos - 1], route[pos])])
        if increase < min_increase:
            min_increase = increase
            best_pos = pos

    return best_pos

# ----------------------------------------------------------------------------------------------------------------------------------------------------------------#

def get_path(graph, start, end):
    queue = deque([[start]])
    visited = set([start])
    while queue:
        path = queue.popleft()
        node = path[-1]
        if node == end:
            return path
        for neighbor in graph.get(node, []):
            if neighbor not in visited:
                visited.add(neighbor)
                new_path = list(path)
                new_path.append(neighbor)
                queue.append(new_path)
    return None

def Get_all_path(route, graph):
    full_path = [route[0]]
    for i in range(1, len(route)):
        segment = get_path(graph, route[i - 1], route[i])
        if segment:
            full_path.extend(segment[1:])
    return full_path

import queue

animation_queue = queue.Queue()
worker_animation_states = {}

global ahahaha
ahahaha = 1

def process_animation_queue():
    global ahahaha
    if not animation_queue.empty():
        task = animation_queue.get()
        task()
        animation_queue.task_done()
    tk.after(ahahaha, process_animation_queue)

tk.after(ahahaha, process_animation_queue)

def go_to(cell1, cell2, current_worker_id, step=0, callback=None):
    if step == 0:
        x1, y1 = convenient_indexes_from[cell1]
        x2, y2 = convenient_indexes_from[cell2]

        center1 = (x1 - Gl / 2, y1 + Gl / 2)
        center2 = (x2 - Gl / 2, y2 + Gl / 2)

        oval_id = solve_workers[current_worker_id]
        coords = reference_canvas.coords(oval_id)

        if not coords:
            current_center = center1
        else:
            current_center = ((coords[0] + coords[2]) / 2, (coords[1] + coords[3]) / 2)

        dx = center2[0] - current_center[0]
        dy = center2[1] - current_center[1]

        worker_animation_states[current_worker_id] = {
            'dx': dx,
            'dy': dy,
            'steps': 20,
            'target_cell': cell2,
            'oval_id': oval_id,
            'callback': callback
        }

    state = worker_animation_states[current_worker_id]
    dx = state['dx']
    dy = state['dy']
    steps = state['steps']
    oval_id = state['oval_id']

    if step < steps:
        reference_canvas.move(oval_id, dx / steps, dy / steps)
        animation_queue.put(lambda: go_to(cell1, cell2, current_worker_id, step + 1, callback))
    else:
        x2, y2 = convenient_indexes_from[state['target_cell']]
        x1_oval = x2 - 2
        y1_oval = y2 + 1
        x2_oval = x2 - Gl + 1
        y2_oval = y2 + Gl - 2
        reference_canvas.coords(oval_id, x1_oval, y1_oval, x2_oval, y2_oval)

        if state['callback']:
            state['callback']()

        del worker_animation_states[current_worker_id]

def Animate(path, current_worker_id):
    segments = []

    for i in range(len(path) - 1):
        segments.append((path[i], path[i + 1]))

    def process_next_segment(segment_index=0):
        if segment_index < len(segments):
            cell1, cell2 = segments[segment_index]

            def next_callback():
                Done(cell2)
                if segment_index < len(segments) - 1:
                    process_next_segment(segment_index + 1)

            animation_queue.put(lambda: go_to(cell1, cell2, current_worker_id, 0, next_callback))

    animation_queue.put(lambda: process_next_segment(0))

def Solve(graph, cheliks, purpose):
    if not cheliks:
        return

    global ahahaha, workers, cells
    if len(workers) == 1 or len(cells) == 1:
        ahahaha = 5
    else:
        ahahaha = 1

    global worker_animation_states
    worker_animation_states = {}

    t = 5
    max_time, routes = solve(graph, cheliks, purpose, t)

    print(f"Максимальное время: {max_time}")
    for i, route in enumerate(routes):
        print(f"Рабочий {i + 1}: {' → '.join(str(p) for p in route)}")
        Animate(Get_all_path(route, graph), i)

#----------------------------------------------------------------------------------------------------------------------------------------------------------------#

tk.mainloop()
