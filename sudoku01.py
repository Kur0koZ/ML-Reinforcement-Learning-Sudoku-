# EX Armaek
import random
from collections import defaultdict

N = 9      # ขนาดกระดาน 9x9
BOX = 3    # ขนาดบล็อกย่อย 3x3 (สี่เหลี่ยมจัตุรัส เหมือน 4x4 เดิม)
EMPTY = 0


class SudokuEnv:
    """Environment สำหรับซูโดกุ 9x9 ในรูปแบบ RL"""

    def __init__(self, puzzle):
        self.start = tuple(puzzle)  # กระดานตั้งต้น (81 ช่อง, 0 = ว่าง)
        self.board = list(self.start)

    def reset(self):
        self.board = list(self.start)
        return tuple(self.board)

    def empty_cells(self):
        return [i for i, v in enumerate(self.board) if v == EMPTY]

    def candidates(self, idx):
        """คืนค่าตัวเลขที่ยังใส่ในช่อง idx ได้โดยไม่ผิดกฎ (ใช้เป็น feature แทนกระดานดิบ)"""
        return [v for v in range(1, N + 1) if self.is_valid_move(idx, v)]

    def mrv_cell(self):
        """เลือกช่องว่างที่มีตัวเลือกน้อยที่สุด (Minimum Remaining Values)
        เพื่อลด action space ต่อสเต็ปให้เล็กลงมาก แทนที่จะให้ agent เลือกทั้ง
        (ช่อง, ค่า) จากช่องว่างทั้งหมดพร้อมกัน -> generalize ข้าม episode ได้ดีขึ้น
        """
        empties = self.empty_cells()
        if not empties:
            return None
        return min(empties, key=lambda i: len(self.candidates(i)))

    def valid_actions(self):
        """คืนค่า action ที่เป็นไปได้ในสเต็ปนี้: (ช่อง MRV, ค่าที่ยังใส่ได้)
        ถ้าช่อง MRV ไม่มีตัวเลือกเหลือเลย แปลว่ากระดานตันแล้ว -> คืนลิสต์ว่าง
        """
        idx = self.mrv_cell()
        if idx is None:
            return []
        return [(idx, v) for v in self.candidates(idx)]

    def is_valid_move(self, idx, val):
        row, col = idx // N, idx % N

        # เช็คแถว
        for c in range(N):
            if self.board[row * N + c] == val:
                return False

        # เช็คคอลัมน์
        for r in range(N):
            if self.board[r * N + col] == val:
                return False

        # เช็คบล็อก 3x3
        br, bc = (row // BOX) * BOX, (col // BOX) * BOX
        for r in range(br, br + BOX):
            for c in range(bc, bc + BOX):
                if self.board[r * N + c] == val:
                    return False

        return True

    def step(self, action):
        idx, val = action
        if self.board[idx] != EMPTY or not self.is_valid_move(idx, val):
            return tuple(self.board), -1, True  # ผิดกฎ -> จบเกม

        self.board[idx] = val

        if EMPTY not in self.board:
            return tuple(self.board), 20, True  # แก้สำเร็จ

        # เช็คทางตัน: ถ้ามีช่องว่างที่ไม่เหลือตัวเลือกให้ใส่แล้ว
        # จบ episode ทันทีแทนที่จะเดินต่อไปเรื่อย ๆ จนกว่าจะชนช่องนั้นเข้าจริง ๆ
        for i in self.empty_cells():
            if not self.candidates(i):
                return tuple(self.board), -5, True  # ตัน -> ไม่มีทางแก้ต่อได้

        return tuple(self.board), 1, False  # เดินถูกกฎ ไปต่อ


class QLearningAgent:
    def __init__(self, alpha=0.3, gamma=0.9, epsilon=0.3):
        self.q = defaultdict(float)   # Q[(state, action)] = ค่า Q
        self.alpha = alpha            # learning rate
        self.gamma = gamma            # discount factor
        self.epsilon = epsilon        # อัตราการสำรวจแบบสุ่ม (exploration)

    def choose_action(self, state, actions):
        if random.random() < self.epsilon:
            return random.choice(actions)  # สำรวจแบบสุ่ม

        # เลือก action ที่ Q สูงสุด (ใช้ประโยชน์จากความรู้เดิม)
        q_values = [self.q[(state, a)] for a in actions]
        max_q = max(q_values)
        best = [a for a, q in zip(actions, q_values) if q == max_q]
        return random.choice(best)

    def update(self, state, action, reward, next_state, next_actions, done):
        current_q = self.q[(state, action)]
        if done or not next_actions:
            target = reward
        else:
            target = reward + self.gamma * max(
                self.q[(next_state, a)] for a in next_actions
            )
        self.q[(state, action)] += self.alpha * (target - current_q)


def print_board(board, title=None):
    """ช่วยพิมพ์กระดาน 9x9 ให้อ่านง่าย"""
    if title:
        print(title)
    for r in range(N):
        row = board[r * N: (r + 1) * N]
        print("  " + " ".join(str(v) if v != EMPTY else "." for v in row))


def train(env, agent, episodes=20000):
    solved_count = 0

    for ep in range(episodes):
        state = env.reset()
        done = False

        while not done:
            # สร้างรายการ Action ที่สามารถเลือกได้ (เฉพาะช่อง MRV และค่าที่ยังใส่ได้จริง)
            actions = env.valid_actions()

            if not actions:
                break

            # Agent เลือก Action
            action = agent.choose_action(state, actions)

            # ทำ Action
            next_state, reward, done = env.step(action)

            # สร้าง Action ของ State ถัดไป
            next_actions = env.valid_actions()

            # Update Q-value
            agent.update(state, action, reward, next_state, next_actions, done)

            state = next_state

            # ตรวจสอบว่าแก้ Sudoku สำเร็จหรือไม่
            if reward == 20:
                solved_count += 1

        # ลดค่า Epsilon
        # จาก Exploration -> Exploitation
        agent.epsilon = max(0.01, agent.epsilon * 0.99998)

        # แสดงความคืบหน้าเป็นระยะ ๆ ระหว่างฝึก
        if (ep + 1) % 5000 == 0:
            print(
                f"  [ฝึกแล้ว {ep + 1}/{episodes} รอบ] "
                f"แก้สำเร็จสะสม {solved_count} ครั้ง, epsilon={agent.epsilon:.3f}"
            )

    print(f"แก้สำเร็จ {solved_count} " f"จาก {episodes} รอบการฝึก")


def solve_with_policy(env, agent, verbose=True):
    """ใช้ policy ที่ฝึกแล้ว (ไม่มีการสุ่ม) เพื่อลองแก้กระดานจริง
    verbose=True จะ print แสดงกระดานเริ่มต้นและทุกสเต็ปที่ agent ตัดสินใจ
    """
    state = env.reset()
    agent.epsilon = 0  # ปิดการสำรวจแบบสุ่ม ใช้ความรู้ล้วน ๆ
    done = False
    steps = 0
    reward = 0

    if verbose:
        print("\n" + "=" * 40)
        print("เริ่มแก้ซูโดกุด้วย policy ที่ฝึกแล้ว")
        print("=" * 40)
        print_board(state, "กระดานเริ่มต้น:")
        print("-" * 40)

    max_steps = len(env.empty_cells()) + 5  # กันวนไม่รู้จบ
    while not done and steps < max_steps:
        actions = env.valid_actions()
        if not actions:
            break

        action = agent.choose_action(state, actions)
        idx, val = action
        row, col = idx // N, idx % N

        state, reward, done = env.step(action)
        steps += 1

        if verbose:
            status = (
                "✅ ถูกกฎ"
                if reward == 1
                else "🏆 กระดานสมบูรณ์!" if reward == 20
                else "🚧 ทางตัน (จบเกม)" if reward == -5
                else "❌ ผิดกฎ (จบเกม)"
            )
            print(
                f"สเต็ปที่ {steps}: เติมค่า {val} ที่ตำแหน่ง (แถว {row}, คอลัมน์ {col}) "
                f"-> reward={reward} [{status}]"
            )
            print_board(state)
            print("-" * 40)

    if verbose:
        print("=" * 40)
        if reward == 20:
            print(f"🎉 แก้สำเร็จภายใน {steps} สเต็ป!")
        else:
            print(
                f"⚠️ ยังไม่สำเร็จหลังจาก {steps} สเต็ป (อาจติดค่าที่ผิดกฎ หรือฝึกไม่พอ)"
            )
        print("=" * 40 + "\n")

    return env.board, reward == 20


if __name__ == "__main__":

    # โจทย์จากรูปที่อัปโหลด (81 ช่อง, ตรวจแล้วว่าไม่มีเลขซ้ำในแถว/คอลัมน์/บล็อก)
    puzzle = [
        4, 3, 0,    5, 0, 0,    0, 0, 6,
        0, 0, 8,    0, 2, 0,    0, 1, 9,
        0, 1, 7,    0, 0, 0,    4, 5, 0,
 
        8, 6, 0,    2, 0, 0,    0, 7, 0,
        0, 7, 4,    8, 0, 0,    2, 6, 0,
        1, 0, 9,    0, 0, 7,    8, 3, 0,
 
        2, 0, 0,    1, 7, 8,    0, 4, 3,
        0, 0, 1,    9, 4, 0,    0, 0, 7,
        0, 0, 0,    6, 5, 0,    0, 0, 0,
    ]

    env = SudokuEnv(puzzle)
    agent = QLearningAgent()

    print("เริ่มฝึก agent ด้วย Q-learning...")
    # ใช้ MRV cell + candidate pruning ทำให้ action space ต่อสเต็ปเล็กลงมาก
    # (จากเดิม 43 ช่อง x 9 ค่า เหลือแค่ตัวเลือกของช่องที่ถูกบีบแคบที่สุด)
    # ทำให้ agent generalize ข้าม episode ได้ดีขึ้นมาก ไม่ต้องใช้รอบฝึกหลักแสน
    train(env, agent, episodes=3000)

    board, solved = solve_with_policy(env, agent, verbose=True)

    print("ผลลัพธ์สุดท้าย:")
    for r in range(N):
        print(board[r * N: (r + 1) * N])
    print("แก้สำเร็จ!" if solved else "ยังแก้ไม่สำเร็จ ลองเพิ่มจำนวนรอบฝึก")
