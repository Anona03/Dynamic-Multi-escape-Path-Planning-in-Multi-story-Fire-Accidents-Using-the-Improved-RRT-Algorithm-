import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import ast, re, sys, math, random, time
import heapq

matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial']
matplotlib.rcParams['axes.unicode_minus'] = False


def load_floor_plan_from_model(model_path='Model.py'):
    try:
        with open(model_path, 'r', encoding='utf-8') as f:
            txt = f.read()
    except FileNotFoundError:
        print(f"警告: 未找到 {model_path}，使用默认 30x30 空地图。")
        return np.zeros((30, 30), dtype=int)

    m = re.search(r'floor_plan\s*=\s*np\.array\s*\(', txt)
    if not m:
        return np.zeros((30, 30), dtype=int)

    start_idx = m.end()
    depth = 1
    i = start_idx
    while i < len(txt) and depth > 0:
        c = txt[i]
        if c == '(':
            depth += 1
        elif c == ')':
            depth -= 1
        i += 1
    if depth != 0:
        raise RuntimeError("解析 floor_plan 时括号不匹配。")
    inner = txt[start_idx:i - 1].strip()
    try:
        python_obj = ast.literal_eval(inner)
    except Exception as e:
        raise RuntimeError(f"解析 floor_plan 内容失败: {e}")
    arr = np.array(python_obj, dtype=int)
    return arr


def extract_fire_params_from_model(model_path='Model.py'):
    try:
        with open(model_path, 'r', encoding='utf-8') as f:
            txt = f.read()
    except:
        return [(15, 15)], 1

    fire_start = []
    m_list = re.search(r'fire_start\s*=\s*(\[.*?\])', txt, re.DOTALL)
    if m_list:
        try:
            fire_start = ast.literal_eval(m_list.group(1))
        except:
            pass

    if not fire_start:
        m_single = re.search(r'fire_start\s*=\s*(\([^\)]+\))', txt)
        if m_single:
            try:
                fire_start = [ast.literal_eval(m_single.group(1))]
            except:
                pass

    if not fire_start:
        fire_start = [(15, 15)]

    fire_speed = 1
    m2 = re.search(r'fire_speed\s*=\s*([0-9]+)', txt)
    if m2:
        try:
            fire_speed = int(m2.group(1))
        except:
            fire_speed = 1

    return fire_start, fire_speed


class Node:
    def __init__(self, x, y, parent=None, index=0, heading=None, time_stamp=0.0):
        self.x = x
        self.y = y
        self.parent = parent
        self.index = index
        self.heading = heading
        self.time_stamp = time_stamp
        self.valid = True


def distance(n1, n2): return math.hypot(n1.x - n2.x, n1.y - n2.y)


def heading_between(n1, n2): return math.atan2(n2.y - n1.y, n2.x - n1.x)


def angle_diff(a, b): return (a - b + math.pi) % (2 * math.pi) - math.pi


def safe_asin(x): return math.asin(max(-1.0, min(1.0, x)))


class PolyObstacle:
    def __init__(self, vertices):
        self.vertices = np.array(vertices)



def is_collision_free_fast(x, y, floor_plan, cell_size=20.0, origin=(0.0, 0.0)):
    col = int((x - origin[0]) // cell_size)
    row_from_bottom = int((y - origin[1]) // cell_size)
    rows, cols = floor_plan.shape
    r = rows - 1 - row_from_bottom

    if 0 <= r < rows and 0 <= col < cols:
        if floor_plan[r, col] == 1:
            return False
        return True
    return False


def grid_to_obstacles(floor_plan, cell_size=20.0, origin=(0.0, 0.0), invert_y=False):

    obstacles = []
    rows, cols = floor_plan.shape
    x0, y0 = origin
    for r in range(rows):
        for c in range(cols):
            if int(floor_plan[r, c]) == 1:
                row_from_bottom = rows - 1 - r if invert_y else r
                x_min = x0 + c * cell_size
                y_min = y0 + row_from_bottom * cell_size
                verts = [(x_min, y_min), (x_min + cell_size, y_min),
                         (x_min + cell_size, y_min + cell_size), (x_min, y_min + cell_size)]
                obstacles.append(PolyObstacle(verts))
    return obstacles


class EnvironmentGrid:
    def __init__(self, floor_plan, fire_start_list, cell_size=20.0, origin=(0.0, 0.0)):
        self.floor_plan = floor_plan.copy()
        self.rows, self.cols = floor_plan.shape
        self.cell_size = cell_size
        self.origin = origin
        self.fire_time_map = np.full((self.rows, self.cols), float('inf'))
        self.smoke_arrival_map = np.full((self.rows, self.cols), float('inf'))
        self.fire_start = fire_start_list if fire_start_list else [(self.rows // 2, self.cols // 2)]
        self.smoke_saturation_time = 30.0

    def _propagate_wavefront(self, start_rc_list, base_interval, max_time, time_map, start_time_offset=0.0,
                             randomness=0.3):
        pq = []
        for (sr, sc) in start_rc_list:
            if 0 <= sr < self.rows and 0 <= sc < self.cols:
                time_map[sr, sc] = start_time_offset
                heapq.heappush(pq, (start_time_offset, sr, sc))

        while pq:
            curr_t, r, c = heapq.heappop(pq)
            if curr_t > time_map[r, c]: continue
            if curr_t >= max_time: continue

            for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                ni, nj = r + di, c + dj
                if 0 <= ni < self.rows and 0 <= nj < self.cols:
                    if self.floor_plan[ni, nj] == 0:
                        noise = random.uniform(-randomness, randomness)
                        step_cost = base_interval * (1.0 + noise)
                        new_time = curr_t + max(0.1, step_cost)
                        if new_time < time_map[ni, nj]:
                            time_map[ni, nj] = new_time
                            heapq.heappush(pq, (new_time, ni, nj))

    def build_maps(self, max_time, fire_interval, smoke_interval, fire_start_time=0.0):
        self._propagate_wavefront(self.fire_start, fire_interval, max_time, self.fire_time_map, fire_start_time,
                                  randomness=0.4)
        self._propagate_wavefront(self.fire_start, smoke_interval, max_time, self.smoke_arrival_map, fire_start_time,
                                  randomness=0.5)

    def get_smoke_concentration(self, r, c, current_time):
        arrival = self.smoke_arrival_map[r, c]
        if current_time < arrival: return 0.0
        conc = (current_time - arrival) / self.smoke_saturation_time
        return min(1.0, max(0.0, conc))

    def is_dangerous(self, x, y, current_time):
        col = int((x - self.origin[0]) // self.cell_size)
        r = self.rows - 1 - int((y - self.origin[1]) // self.cell_size)
        if 0 <= r < self.rows and 0 <= col < self.cols:
            if current_time >= self.fire_time_map[r, col]: return True
            if self.get_smoke_concentration(r, col, current_time) > 0.5: return True
        return False

    def get_active_cells_at_time(self, t):
        cells = []
        for r in range(self.rows):
            for c in range(self.cols):
                if t >= self.fire_time_map[r, c]:
                    cells.append((r, c, 'fire', 1.0))
                else:
                    conc = self.get_smoke_concentration(r, c, t)
                    if conc > 0.05: cells.append((r, c, 'smoke', conc))
        return cells


# 上一层楼
def rrt_star_smoke_multigoal(model_path='Model_fire.py', start_cell=None, tasks_config=None,
                             cell_size=20.0, origin=(0.0, 0.0), no_of_iterations=3000, step_len=30.0,
                             robot_speed=40.0, fire_spread_interval=4.0, smoke_spread_interval=1.5,
                             fire_max_time=1000.0, fire_start_time=0.0, MAX_PATHS=1,
                             min_turn_radius=25.0, max_turn_radius=None, use_heading_constraints=True):
    floor_plan = load_floor_plan_from_model(model_path)
    fire_start, _ = extract_fire_params_from_model(model_path)

    env_grid = EnvironmentGrid(floor_plan, fire_start, cell_size, origin)
    print(f"[上层] 正在预计算火势与烟雾扩散 (Max Time: {fire_max_time}s, 起火时间: {fire_start_time}s)...")
    env_grid.build_maps(max_time=fire_max_time, fire_interval=fire_spread_interval,
                        smoke_interval=smoke_spread_interval, fire_start_time=fire_start_time)

    static_obstacles = grid_to_obstacles(floor_plan, cell_size=cell_size, origin=origin, invert_y=True)
    rows, cols = floor_plan.shape
    x_min, y_min = origin
    x_max, y_max = origin[0] + cols * cell_size, origin[1] + rows * cell_size

    boundary_margin = 0.25 * cell_size
    eff_x_min, eff_y_min = x_min + boundary_margin, y_min + boundary_margin
    eff_x_max, eff_y_max = x_max - boundary_margin, y_max - boundary_margin

    if start_cell is None: start_cell = (rows - 1, 0)

    def cell_center(rc):
        return (origin[0] + rc[1] * cell_size + cell_size * 0.5,
                origin[1] + (rows - 1 - rc[0]) * cell_size + cell_size * 0.5)

    Sxy = cell_center(start_cell)

    tasks = []
    for idx, cfg in enumerate(tasks_config):
        gx, gy = cell_center(cfg['goal_cell'])
        goal_node = Node(gx, gy, heading=None, time_stamp=0.0)
        start_heading = math.atan2(goal_node.y - Sxy[1], goal_node.x - Sxy[0])
        start_node_copy = Node(Sxy[0], Sxy[1], heading=start_heading, time_stamp=0.0)

        task_data = {'id': idx, 'config': cfg, 'goal_node': goal_node, 'tree': [start_node_copy], 'paths': [],
                     'rect_params': None, 'line_collection': [], 'color': 'blue' if idx == 0 else 'purple',
                     'goal_color': 'orangered' if idx == 0 else 'gold'}

        vx, vy = goal_node.x - start_node_copy.x, goal_node.y - start_node_copy.y
        dist = math.hypot(vx, vy)
        if dist > 1e-6:
            ux, uy = vx / dist, vy / dist
            px, py = -uy, ux
            width = cfg.get('rect_w_factor', 5.0) * cell_size
            length_total = dist * max(1e-3, cfg.get('rect_l_factor', 1.2))
            extra_len = (length_total - dist) / 2.0
            off_real = cfg.get('rect_offset', 0.0) * cell_size
            base_cx = start_node_copy.x - ux * extra_len + px * off_real
            base_cy = start_node_copy.y - uy * extra_len + py * off_real
            half_w = width / 2.0
            p0 = (base_cx - px * half_w, base_cy - py * half_w)
            p1 = (base_cx + px * half_w, base_cy + py * half_w)
            p2 = (p1[0] + ux * length_total, p1[1] + uy * length_total)
            p3 = (p0[0] + ux * length_total, p0[1] + uy * length_total)
            task_data['rect_params'] = {'poly_points': [p0, p1, p2, p3], 'base_center': (base_cx, base_cy),
                                        'ux': ux, 'uy': uy, 'px': px, 'py': py, 'length': length_total, 'width': width}
        tasks.append(task_data)

    r_find = 2.0 * step_len
    connection_threshold = step_len * 1.5
    alphy_min, alphy_max = 0.0, math.pi
    if use_heading_constraints and min_turn_radius is not None:
        alphy_max = math.pi if step_len >= 2 * min_turn_radius else 2 * safe_asin(step_len / (2 * min_turn_radius))

    def heading_ok(parent_heading, new_heading):
        if parent_heading is None: return True
        delta = abs(angle_diff(new_heading, parent_heading))
        return (delta + 1e-6 >= alphy_min) and (delta - 1e-6 <= alphy_max)


    def is_collision_free_time_line(n1, n2, num_points=10):
        dt = n2.time_stamp - n1.time_stamp
        for i in range(num_points + 1):
            ratio = i / num_points
            x, y = n1.x + (n2.x - n1.x) * ratio, n1.y + (n2.y - n1.y) * ratio
            t_curr = n1.time_stamp + dt * ratio
            if not is_collision_free_fast(x, y, floor_plan, cell_size, origin): return False
            if env_grid.is_dangerous(x, y, t_curr): return False
        return True

    def get_sample_for_task(task):
        cfg = task['config']
        rect = task['rect_params']
        if rect is None or random.random() < cfg.get('goal_prob', 0.1):
            return Node(random.uniform(eff_x_min, eff_x_max), random.uniform(eff_y_min, eff_y_max))
        u, v = random.random(), max(-cfg.get('gauss_clip', 0.5),
                                    min(cfg.get('gauss_clip', 0.5), random.gauss(0.0, cfg.get('gauss_sigma', 0.15))))
        x = rect['base_center'][0] + rect['ux'] * rect['length'] * u + rect['px'] * rect['width'] * v
        y = rect['base_center'][1] + rect['uy'] * rect['length'] * u + rect['py'] * rect['width'] * v
        return Node(min(max(x, eff_x_min), eff_x_max), min(max(y, eff_y_min), eff_y_max))

    def expand_task_tree(task, rnd_node):
        tree_nodes = task['tree']
        valid_nodes = [n for n in tree_nodes if n.valid]
        if not valid_nodes: return None
        nearest = min(valid_nodes, key=lambda n: math.hypot(n.x - rnd_node.x, n.y - rnd_node.y))
        theta = math.atan2(rnd_node.y - nearest.y, rnd_node.x - nearest.x)
        new_x, new_y = min(max(nearest.x + step_len * math.cos(theta), eff_x_min), eff_x_max), min(
            max(nearest.y + step_len * math.sin(theta), eff_y_min), eff_y_max)
        temp_time = nearest.time_stamp + math.hypot(new_x - nearest.x, new_y - nearest.y) / robot_speed
        if env_grid.is_dangerous(new_x, new_y, temp_time): return None

        new_node = Node(new_x, new_y, index=len(tree_nodes))
        near_indices = []
        for i, node in enumerate(tree_nodes):
            if node.valid and distance(new_node, node) <= r_find:
                t_new = node.time_stamp + distance(new_node, node) / robot_speed
                if not env_grid.is_dangerous(new_node.x, new_node.y, t_new) and is_collision_free_time_line(node,
                                                                                                            Node(
                                                                                                                new_node.x,
                                                                                                                new_node.y,
                                                                                                                time_stamp=t_new)):
                    if heading_ok(node.heading, heading_between(node, new_node)): near_indices.append(i)
        if not near_indices: return None

        min_time, best_parent = float('inf'), -1
        for idx in near_indices:
            arrival_time = tree_nodes[idx].time_stamp + distance(tree_nodes[idx], new_node) / robot_speed
            if arrival_time < min_time:
                min_time, best_parent, best_heading = arrival_time, idx, heading_between(tree_nodes[idx], new_node)

        if best_parent == -1: return None
        new_node.parent, new_node.heading, new_node.time_stamp = best_parent, best_heading, min_time
        tree_nodes.append(new_node)
        parent_node = tree_nodes[new_node.parent]
        ln, = ax.plot([parent_node.x, new_node.x], [parent_node.y, new_node.y], '-', color=task['color'], alpha=0.5,
                      linewidth=0.8)
        task['line_collection'].append(ln)
        return new_node

    plt.ion()
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_xlim(x_min - cell_size, x_max + cell_size)
    ax.set_ylim(y_min - cell_size, y_max + cell_size)
    plt.title("[Upper Floor] Multi-Goal RRT*")

    for obs in static_obstacles: ax.add_patch(
        patches.Rectangle(obs.vertices[0], cell_size, cell_size, color='k', alpha=0.8))
    ax.plot(Sxy[0], Sxy[1], 'o', color='lime', markersize=8, label='Start')

    for t in tasks:
        ax.plot(t['goal_node'].x, t['goal_node'].y, 'o', color=t['goal_color'], markersize=8,
                label=f"Goal {t['id'] + 1}")
        rp = t['rect_params']
        if rp:
            poly = patches.Polygon(rp['poly_points'], closed=True, edgecolor=t['color'], facecolor='none', alpha=0.3,
                                   linestyle='--', linewidth=1.5)
            ax.add_patch(poly)

    smoke_legend = patches.Patch(color='gray', alpha=0.5, label='Smoke (>50% Block)')
    fire_legend = patches.Patch(color='red', alpha=0.6, label='Fire')
    ax.legend(handles=[smoke_legend, fire_legend])

    hazard_patches = []
    task_arrival_times = {}
    print(f"开始多目标规划...")

    for it in range(no_of_iterations):
        all_done = True

        # 【性能优化】：将渲染频率从 10 降低到 50
        if it % 50 == 0:
            for p in hazard_patches: p.remove()
            hazard_patches.clear()
            ref_time = max([t['tree'][-1].time_stamp for t in tasks if t['tree']] + [0.0])
            for r, c, type_, val in env_grid.get_active_cells_at_time(ref_time):
                x0, y0 = origin[0] + c * cell_size, origin[1] + (rows - 1 - r) * cell_size
                if type_ == 'fire':
                    rp = patches.Rectangle((x0, y0), cell_size, cell_size, color='red', alpha=0.6)
                else:
                    rp = patches.Rectangle((x0, y0), cell_size, cell_size, facecolor='gray',
                                           alpha=min(0.9, 0.1 + 0.8 * val), edgecolor='black' if val > 0.5 else 'none')
                ax.add_patch(rp)
                hazard_patches.append(rp)

        for t in tasks:
            if len(t['paths']) < MAX_PATHS:
                all_done = False
                new_node = expand_task_tree(t, get_sample_for_task(t))
                if new_node:
                    d_to_goal = distance(new_node, t['goal_node'])
                    if d_to_goal <= connection_threshold:
                        t_arrival = new_node.time_stamp + d_to_goal / robot_speed
                        if not env_grid.is_dangerous(t['goal_node'].x, t['goal_node'].y, t_arrival):
                            if is_collision_free_time_line(new_node, Node(t['goal_node'].x, t['goal_node'].y,
                                                                          time_stamp=t_arrival)):
                                if heading_ok(new_node.heading, heading_between(new_node, t['goal_node'])):
                                    final_gn = Node(t['goal_node'].x, t['goal_node'].y, parent=new_node.index,
                                                    time_stamp=t_arrival, index=len(t['tree']))
                                    t['tree'].append(final_gn)
                                    path = []
                                    curr = final_gn
                                    while curr:
                                        path.append((curr.x, curr.y))
                                        if curr.parent is None: break
                                        curr = t['tree'][curr.parent]
                                    path.reverse()
                                    path_arr = np.array(path)
                                    print(f"[Iter {it}] Task {t['id'] + 1} 到达上层终点! 时间: {t_arrival:.2f}s")
                                    t['paths'].append(path_arr)
                                    task_arrival_times[t['id']] = t_arrival
                                    ax.plot(path_arr[:, 0], path_arr[:, 1], '-', color='lime', linewidth=3, alpha=0.9)
        if all_done: break

        # 【性能优化】： UI 刷新率从 5 改为 50
        if it % 50 == 0: plt.pause(0.001)

    print(f"上层规划结束。画面将停留 3 秒钟，然后自动切换至下一层...")
    plt.pause(3.0)
    plt.close(fig)
    return task_arrival_times, tasks, it + 1


# 下一层楼
def rrt_star_smoke_multistart(model_path='Model_fire2.py', goal_cell=None, agents_config=None,
                              cell_size=20.0, origin=(0.0, 0.0), no_of_iterations=3000, step_len=30.0,
                              robot_speed=40.0, fire_spread_interval=4.0, smoke_spread_interval=1.5,
                              fire_max_time=1000.0, fire_start_time=0.0, MAX_PATHS=1,
                              min_turn_radius=25.0, max_turn_radius=None, use_heading_constraints=True):
    floor_plan = load_floor_plan_from_model(model_path)
    fire_start, _ = extract_fire_params_from_model(model_path)

    env_grid = EnvironmentGrid(floor_plan, fire_start, cell_size, origin)
    print(f"[下层] 正在预计算火势与烟雾扩散 (Max Time: {fire_max_time}s, 起火时间: {fire_start_time}s)...")
    env_grid.build_maps(max_time=fire_max_time, fire_interval=fire_spread_interval,
                        smoke_interval=smoke_spread_interval, fire_start_time=fire_start_time)

    static_obstacles = grid_to_obstacles(floor_plan, cell_size=cell_size, origin=origin, invert_y=True)
    rows, cols = floor_plan.shape
    x_min, y_min = origin
    x_max, y_max = origin[0] + cols * cell_size, origin[1] + rows * cell_size

    boundary_margin = 0.25 * cell_size
    eff_x_min, eff_y_min = x_min + boundary_margin, y_min + boundary_margin
    eff_x_max, eff_y_max = x_max - boundary_margin, y_max - boundary_margin

    def cell_center(rc):
        return (origin[0] + rc[1] * cell_size + cell_size * 0.5,
                origin[1] + (rows - 1 - rc[0]) * cell_size + cell_size * 0.5)

    if goal_cell is None: goal_cell = (0, cols - 1)
    Gxy = cell_center(goal_cell)
    common_goal_node = Node(Gxy[0], Gxy[1])

    tasks = []
    for idx, cfg in enumerate(agents_config):
        Sxy = cell_center(cfg['start_cell'])
        start_heading = math.atan2(common_goal_node.y - Sxy[1], common_goal_node.x - Sxy[0])
        agent_start_time = cfg.get('start_time', 0.0)
        start_node_copy = Node(Sxy[0], Sxy[1], heading=start_heading, time_stamp=agent_start_time)

        task_data = {'id': idx, 'config': cfg, 'start_node': start_node_copy, 'goal_node': Node(Gxy[0], Gxy[1]),
                     'tree': [start_node_copy], 'paths': [], 'rect_params': None, 'line_collection': [],
                     'color': 'blue' if idx == 0 else 'purple', 'start_color': 'lime' if idx == 0 else 'cyan'}

        vx, vy = common_goal_node.x - start_node_copy.x, common_goal_node.y - start_node_copy.y
        dist = math.hypot(vx, vy)
        if dist > 1e-6:
            ux, uy = vx / dist, vy / dist
            px, py = -uy, ux
            width = cfg.get('rect_w_factor', 5.0) * cell_size
            length_total = dist * max(1e-3, cfg.get('rect_l_factor', 1.2))
            extra_len = (length_total - dist) / 2.0
            off_real = cfg.get('rect_offset', 0.0) * cell_size
            base_cx = start_node_copy.x - ux * extra_len + px * off_real
            base_cy = start_node_copy.y - uy * extra_len + py * off_real
            half_w = width / 2.0
            p0 = (base_cx - px * half_w, base_cy - py * half_w)
            p1 = (base_cx + px * half_w, base_cy + py * half_w)
            p2 = (p1[0] + ux * length_total, p1[1] + uy * length_total)
            p3 = (p0[0] + ux * length_total, p0[1] + uy * length_total)
            task_data['rect_params'] = {'poly_points': [p0, p1, p2, p3], 'base_center': (base_cx, base_cy),
                                        'ux': ux, 'uy': uy, 'px': px, 'py': py, 'length': length_total, 'width': width}
        tasks.append(task_data)

    r_find = 2.0 * step_len
    connection_threshold = step_len * 1.5
    alphy_min, alphy_max = 0.0, math.pi
    if use_heading_constraints and min_turn_radius is not None:
        alphy_max = math.pi if step_len >= 2 * min_turn_radius else 2 * safe_asin(step_len / (2 * min_turn_radius))

    def heading_ok(parent_heading, new_heading):
        if parent_heading is None: return True
        delta = abs(angle_diff(new_heading, parent_heading))
        return (delta + 1e-6 >= alphy_min) and (delta - 1e-6 <= alphy_max)

    # 【核心修改】：下层 O(1) 连线检测
    def is_collision_free_time_line(n1, n2, num_points=10):
        dt = n2.time_stamp - n1.time_stamp
        for i in range(num_points + 1):
            ratio = i / num_points
            x, y = n1.x + (n2.x - n1.x) * ratio, n1.y + (n2.y - n1.y) * ratio
            t_curr = n1.time_stamp + dt * ratio
            if not is_collision_free_fast(x, y, floor_plan, cell_size, origin): return False
            if env_grid.is_dangerous(x, y, t_curr): return False
        return True

    def get_sample_for_task(task):
        cfg = task['config']
        rect = task['rect_params']
        if rect is None or random.random() < cfg.get('goal_prob', 0.1):
            return Node(random.uniform(eff_x_min, eff_x_max), random.uniform(eff_y_min, eff_y_max))
        u, v = random.random(), max(-cfg.get('gauss_clip', 0.5),
                                    min(cfg.get('gauss_clip', 0.5), random.gauss(0.0, cfg.get('gauss_sigma', 0.15))))
        x = rect['base_center'][0] + rect['ux'] * rect['length'] * u + rect['px'] * rect['width'] * v
        y = rect['base_center'][1] + rect['uy'] * rect['length'] * u + rect['py'] * rect['width'] * v
        return Node(min(max(x, eff_x_min), eff_x_max), min(max(y, eff_y_min), eff_y_max))

    def expand_task_tree(task, rnd_node):
        tree_nodes = task['tree']
        valid_nodes = [n for n in tree_nodes if n.valid]
        if not valid_nodes: return None
        nearest = min(valid_nodes, key=lambda n: math.hypot(n.x - rnd_node.x, n.y - rnd_node.y))
        theta = math.atan2(rnd_node.y - nearest.y, rnd_node.x - nearest.x)
        new_x, new_y = min(max(nearest.x + step_len * math.cos(theta), eff_x_min), eff_x_max), min(
            max(nearest.y + step_len * math.sin(theta), eff_y_min), eff_y_max)
        temp_time = nearest.time_stamp + math.hypot(new_x - nearest.x, new_y - nearest.y) / robot_speed
        if env_grid.is_dangerous(new_x, new_y, temp_time): return None

        new_node = Node(new_x, new_y, index=len(tree_nodes))
        near_indices = []
        for i, node in enumerate(tree_nodes):
            if node.valid and distance(new_node, node) <= r_find:
                t_new = node.time_stamp + distance(new_node, node) / robot_speed
                if not env_grid.is_dangerous(new_node.x, new_node.y, t_new) and is_collision_free_time_line(node,
                                                                                                            Node(
                                                                                                                new_node.x,
                                                                                                                new_node.y,
                                                                                                                time_stamp=t_new)):
                    if heading_ok(node.heading, heading_between(node, new_node)): near_indices.append(i)
        if not near_indices: return None

        min_time, best_parent = float('inf'), -1
        for idx in near_indices:
            arrival_time = tree_nodes[idx].time_stamp + distance(tree_nodes[idx], new_node) / robot_speed
            if arrival_time < min_time:
                min_time, best_parent, best_heading = arrival_time, idx, heading_between(tree_nodes[idx], new_node)

        if best_parent == -1: return None
        new_node.parent, new_node.heading, new_node.time_stamp = best_parent, best_heading, min_time
        tree_nodes.append(new_node)
        parent_node = tree_nodes[new_node.parent]
        ln, = ax.plot([parent_node.x, new_node.x], [parent_node.y, new_node.y], '-', color=task['color'], alpha=0.5,
                      linewidth=0.8)
        task['line_collection'].append(ln)
        return new_node

    plt.ion()
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_xlim(x_min - cell_size, x_max + cell_size)
    ax.set_ylim(y_min - cell_size, y_max + cell_size)
    plt.title("[Lower Floor] Multi-Start RRT* with Inherited Time")

    for obs in static_obstacles: ax.add_patch(
        patches.Rectangle(obs.vertices[0], cell_size, cell_size, color='k', alpha=0.8))
    ax.plot(common_goal_node.x, common_goal_node.y, 'o', color='orangered', markersize=12, label='Common Goal')

    for t in tasks:
        ax.plot(t['start_node'].x, t['start_node'].y, 'o', color=t['start_color'], markersize=8,
                label=f"Start {t['id'] + 1} (T={t['start_node'].time_stamp:.1f}s)")
        rp = t['rect_params']
        if rp:
            poly = patches.Polygon(rp['poly_points'], closed=True, edgecolor=t['color'], facecolor='none', alpha=0.3,
                                   linestyle='--', linewidth=1.5)
            ax.add_patch(poly)

    smoke_legend = patches.Patch(color='gray', alpha=0.5, label='Smoke (>50% Block)')
    fire_legend = patches.Patch(color='red', alpha=0.6, label='Fire')
    ax.legend(handles=[smoke_legend, fire_legend])

    hazard_patches = []
    print(f"开始多起点规划...")

    for it in range(no_of_iterations):
        all_done = True

        # 【性能优化】：将渲染频率从 10 降低到 50
        if it % 50 == 0:
            for p in hazard_patches: p.remove()
            hazard_patches.clear()
            ref_time = max([t['tree'][-1].time_stamp for t in tasks if t['tree']] + [0.0])
            for r, c, type_, val in env_grid.get_active_cells_at_time(ref_time):
                x0, y0 = origin[0] + c * cell_size, origin[1] + (rows - 1 - r) * cell_size
                if type_ == 'fire':
                    rp = patches.Rectangle((x0, y0), cell_size, cell_size, color='red', alpha=0.6)
                else:
                    rp = patches.Rectangle((x0, y0), cell_size, cell_size, facecolor='gray',
                                           alpha=min(0.9, 0.1 + 0.8 * val), edgecolor='black' if val > 0.5 else 'none')
                ax.add_patch(rp)
                hazard_patches.append(rp)

        for t in tasks:
            if len(t['paths']) < MAX_PATHS:
                all_done = False
                new_node = expand_task_tree(t, get_sample_for_task(t))
                if new_node:
                    d_to_goal = distance(new_node, t['goal_node'])
                    if d_to_goal <= connection_threshold:
                        t_arrival = new_node.time_stamp + d_to_goal / robot_speed
                        if not env_grid.is_dangerous(t['goal_node'].x, t['goal_node'].y, t_arrival):
                            if is_collision_free_time_line(new_node, Node(t['goal_node'].x, t['goal_node'].y,
                                                                          time_stamp=t_arrival)):
                                if heading_ok(new_node.heading, heading_between(new_node, t['goal_node'])):
                                    final_gn = Node(t['goal_node'].x, t['goal_node'].y, parent=new_node.index,
                                                    time_stamp=t_arrival, index=len(t['tree']))
                                    t['tree'].append(final_gn)
                                    path = []
                                    curr = final_gn
                                    while curr:
                                        path.append((curr.x, curr.y))
                                        if curr.parent is None: break
                                        curr = t['tree'][curr.parent]
                                    path.reverse()
                                    path_arr = np.array(path)
                                    print(f"[Iter {it}] Agent {t['id'] + 1} 到达最终出口! 时间: {t_arrival:.2f}s")
                                    t['paths'].append(path_arr)
                                    ax.plot(path_arr[:, 0], path_arr[:, 1], '-',
                                            color='lime' if t['id'] == 0 else 'magenta', linewidth=3, alpha=0.9)
        if all_done: break


        if it % 50 == 0: plt.pause(0.001)

    print(f"下层规划结束。")
    plt.ioff()
    plt.show()
    return tasks, it + 1


if __name__ == "__main__":
    UPPER_FIRE_START = 0.0
    LOWER_FIRE_START = 3.0
    STAIR_TRANSIT_TIME = 5.0

    task1_config = {'goal_cell': (0, 18), 'rect_w_factor': 22.0, 'rect_l_factor': 1.2, 'rect_offset': 0.0,
                    'gauss_sigma': 0.15, 'gauss_clip': 0.5, 'goal_prob': 0.1}
    task2_config = {'goal_cell': (24, 29), 'rect_w_factor': 18.0, 'rect_l_factor': 1.5, 'rect_offset': 0.0,
                    'gauss_sigma': 0.3, 'gauss_clip': 0.8, 'goal_prob': 0.1}

    arrival_times, upper_tasks, upper_iterations = rrt_star_smoke_multigoal(
        model_path='Model_fire.py',
        start_cell=(29, 0),
        tasks_config=[task1_config, task2_config],
        no_of_iterations=1000, step_len=30.0, robot_speed=13.0,
        fire_spread_interval=20.0, smoke_spread_interval=6.7, fire_max_time=2000.0,
        fire_start_time=UPPER_FIRE_START, MAX_PATHS=1
    )

    agent1_start_time = arrival_times.get(0, 0.0) + STAIR_TRANSIT_TIME
    agent2_start_time = arrival_times.get(1, 0.0) + STAIR_TRANSIT_TIME

    print(
        f"\n跨楼层时间同步：Agent 1 将在 T={agent1_start_time:.2f}s 开始下一层规划，Agent 2 将在 T={agent2_start_time:.2f}s 开始。")

    agent1_config = {'start_cell': (0, 18), 'start_time': agent1_start_time, 'rect_w_factor': 22.0,
                     'rect_l_factor': 1.2, 'rect_offset': 0.0, 'gauss_sigma': 0.15, 'gauss_clip': 0.5, 'goal_prob': 0.1}
    agent2_config = {'start_cell': (24, 29), 'start_time': agent2_start_time, 'rect_w_factor': 18.0,
                     'rect_l_factor': 1.5, 'rect_offset': 0.0, 'gauss_sigma': 0.3, 'gauss_clip': 0.8, 'goal_prob': 0.1}

    lower_tasks, lower_iterations = rrt_star_smoke_multistart(
        model_path='Model_fire2.py',
        goal_cell=(29, 5),
        agents_config=[agent1_config, agent2_config],
        no_of_iterations=1000, step_len=30.0, robot_speed=13.0,
        fire_spread_interval=20.0, smoke_spread_interval=6.7, fire_max_time=2000.0,
        fire_start_time=LOWER_FIRE_START, MAX_PATHS=1
    )