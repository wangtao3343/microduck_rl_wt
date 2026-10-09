"""把 MJCF 里的 XL330 质量换成实测的 HD-1910 质量。

## 为什么

实机（含电池、**不含**头部电子件）实称 **841.2 g**，仿真 **737.24 g**，差 **104.0 g**。
实称一颗装好的 HD-1910（含舵盘含线）**24.6 g**，仿真里那颗 XL330 是 18 g：

    15 × (24.6 − 18) = +99.0 g   ← 舵机
    104.0 − 99.0     =  +5.0 g   ← 结构件残差（≈0，说明 CAD 结构是对的）

也就是说 **841.2 g 里没有装那 44.9 g 电子件**（Radxa 15.2 + 飞特总线板/降压 29.7），
它们要另外算。头部电子件按用户方案装在头（`jaw_soft`）里。

## 做了什么

对每个身体，按**点质量**模型精确重算（不是简单按比例缩放）：

    m_s  = m0 − Σ old_i                    # 剥出结构件质量
    c_s  = (m0·c0 − Σ old_i·p_i) / m_s     # 结构件质心
    I_s  = I0 − m_s·PA(c_s−c0) − Σ old_i·PA(p_i−c0)
    m1   = m_s + Σ new_i
    c1   = (m_s·c_s + Σ new_i·p_i) / m1    # 新质心
    I1   = I_s + m_s·PA(c_s−c1) + Σ new_i·PA(p_i−c1)

其中 PA(v) = |v|²·𝟙 − v·vᵀ 是平行轴项。舵机/电子件当作点质量（用网格的 `pos`，
onshape-to-robot 导出的零件原点 —— 对近似对称的舵机就是它的质心）。

**幂等**：改之前先断言当前质量等于 XL330 原值；重复运行会报错，不会叠加。

用法：
    uv run scripts/set_hd1910_masses.py            # 改
    uv run scripts/set_hd1910_masses.py --check    # 只看会改成什么
"""
from __future__ import annotations

import argparse
import glob
import os
import re

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(ROOT, "src/mjlab_microduck/robot/microduck")

# ── 实称数据 ────────────────────────────────────────────────────────────────
SERVO_OLD_G = 18.0      # 仿真里那颗 XL330（Dynamixel 官方零件，实重 18 g）
SERVO_NEW_G = 24.6      # HD-1910-C001 实称，含舵盘、含线（裸舵机 22.8）
RADXA_G = 15.2          # 实称
BUSBOARD_G = 29.7       # 飞特官方总线板 + 降压模块，实称
# CAD 里那两块板的质量：网格体积 × 密度。密度由 xl330 零件标定
# （15.68 cm³ ↔ 18 g → 1.148 g/cm³），Pi Zero 3.02 cm³、HAT 1.57 cm³。
# ⚠️ 这是估计值，±5 g；实机那 44.9 g 是实称。
PI_ZERO_CAD_G = 3.47
HAT_CAD_G = 1.80

# 每个身体要替换的点质量：(网格名, 旧质量 g, 新质量 g)。None 键 = 所有身体通用。
POINT_MASS_RULES = {
    None: [("xl330", SERVO_OLD_G, SERVO_NEW_G)],
    # 头部：CAD 的 Pi Zero + HAT 换成实机的 Radxa + 飞特总线板/降压模块
    "jaw_soft": [
        ("xl330", SERVO_OLD_G, SERVO_NEW_G),
        ("pcb__raspberry_pi_zero_2_w", PI_ZERO_CAD_G, RADXA_G),
        ("elec_rpi_robot_hat_pcb", HAT_CAD_G, BUSBOARD_G),
    ],
}

# XL330 原值，用来断言没跑过第二遍
ORIGINAL_G = {
    "trunk_base": 0.199224, "yaw2roll": 0.0230406, "hip_l": 0.00618934,
    "upper_leg_left": 0.0482067, "leg": 0.0215844, "ankle_left": 0.0300246,
    "neck": 0.0368414, "neck_pitch": 0.00572, "yaw_roll_motion": 0.0486,
    "jaw_soft": 0.188766, "bearing_roll": 0.0230406, "hip_l_2": 0.00618934,
    "upper_leg_right": 0.0482067, "leg_2": 0.0215844, "ankle_right": 0.0300251,
}

BODY_RE = re.compile(r'<body name="([^"]+)"')
INERTIAL_RE = re.compile(
    r'<inertial pos="([^"]+)" mass="([^"]+)" fullinertia="([^"]+)"'
)
GEOM_RE = re.compile(r'<geom type="mesh" class="\w+" pos="([^"]+)" quat="[^"]+" mesh="([^"]+)"')


def vec(s: str) -> np.ndarray:
    return np.array([float(x) for x in s.split()])


def pa(v: np.ndarray) -> np.ndarray:
    """平行轴项 |v|²·𝟙 − v·vᵀ。"""
    return np.dot(v, v) * np.eye(3) - np.outer(v, v)


def sym(i: np.ndarray) -> list[float]:
    """对称阵 → MJCF 的 [Ixx Iyy Izz Ixy Ixz Iyz]。"""
    return [i[0, 0], i[1, 1], i[2, 2], i[0, 1], i[0, 2], i[1, 2]]


def body_spans(lines: list[str]) -> list[tuple[str, int, int]]:
    """(body 名, 起始行, 结束行)。用栈处理嵌套。"""
    out, stack = [], []
    for i, ln in enumerate(lines):
        m = BODY_RE.search(ln)
        if m:
            stack.append((m.group(1), i))
        if "</body>" in ln and stack:
            name, start = stack.pop()
            out.append((name, start, i))
    return out


def points_for(body: str, lines: list[str], start: int, end: int) -> list[tuple[np.ndarray, float, float]]:
    """收集该 body 自己的点质量（在第一个子 body 之前），按 (网格,位置) 去重。

    allcollisions 变体里同一颗舵机有 visual + collision 两份 geom，位置相同 → 去重。
    """
    rules = POINT_MASS_RULES[None] + POINT_MASS_RULES.get(body, [])
    by_mesh = {mesh: (old, new) for mesh, old, new in rules}
    pts, seen = [], set()
    for ln in lines[start + 1:end]:
        if BODY_RE.search(ln):          # 进到子 body 了，停
            break
        m = GEOM_RE.search(ln)
        if not m:
            continue
        pos_s, mesh = m.group(1), m.group(2)
        if mesh not in by_mesh:
            continue
        key = (mesh, tuple(round(float(x), 6) for x in pos_s.split()))
        if key in seen:
            continue
        seen.add(key)
        old, new = by_mesh[mesh]
        pts.append((vec(pos_s), old / 1000.0, new / 1000.0))
    return pts


def transform(body: str, old_mass: float, old_pos: str, old_inertia: str,
              pts: list[tuple[np.ndarray, float, float]]) -> tuple[float, np.ndarray, np.ndarray]:
    """按点质量模型重算 (质量, 质心, 惯量阵)。"""
    m0, c0 = old_mass, vec(old_pos)
    i6 = vec(old_inertia)
    i0 = np.array([[i6[0], i6[3], i6[4]],
                   [i6[3], i6[1], i6[5]],
                   [i6[4], i6[5], i6[2]]])

    m_s = m0 - sum(o for _, o, _ in pts)
    if m_s <= 0:
        raise ValueError(f"{body}: 结构件质量为负 ({m_s:.6f})，实称质量对不上模型")
    c_s = (m0 * c0 - sum(o * p for p, o, _ in pts)) / m_s
    i_s = i0 - m_s * pa(c_s - c0) - sum(o * pa(p - c0) for p, o, _ in pts)

    m1 = m_s + sum(n for _, _, n in pts)
    c1 = (m_s * c_s + sum(n * p for p, _, n in pts)) / m1
    i1 = i_s + m_s * pa(c_s - c1) + sum(n * pa(p - c1) for p, _, n in pts)

    if np.any(np.linalg.eigvalsh(i1) <= 0):
        raise ValueError(f"{body}: 重算出的惯量不是正定，检查舵机位置")
    return m1, c1, i1


def process(path: str, check: bool, verbose: bool) -> float:
    lines = open(path).read().split("\n")
    changed, before, after = 0, 0.0, 0.0

    for body, start, end in body_spans(lines):
        inertial_idx = None
        for i in range(start + 1, end):
            if BODY_RE.search(lines[i]):
                break
            if INERTIAL_RE.search(lines[i]):
                inertial_idx = i
                break
        if inertial_idx is None:
            continue

        m = INERTIAL_RE.search(lines[inertial_idx])
        old_mass = float(m.group(2))
        before += old_mass

        if body in ORIGINAL_G:
            assert abs(old_mass - ORIGINAL_G[body]) < 1e-9, (
                f"{os.path.basename(path)} {body}: 当前质量 {old_mass} 不等于 XL330 原值 "
                f"{ORIGINAL_G[body]} —— 这个文件已经改过了，别跑第二遍"
            )

        pts = points_for(body, lines, start, end)
        if not pts:
            after += old_mass
            continue

        m1, c1, i1 = transform(body, old_mass, m.group(1), m.group(3), pts)
        after += m1
        changed += 1

        if verbose or check:
            old_in = vec(m.group(3))
            print(f"  {body:<18s} {old_mass*1000:7.2f} → {m1*1000:7.2f} g   "
                  f"({len(pts)} 点)   质心 Δ={(c1 - vec(m.group(1)))*1000} mm   "
                  f"Ixx {old_in[0]:.3e} → {i1[0,0]:.3e}")

        if not check:
            new_line = INERTIAL_RE.sub(
                '<inertial pos="{:s}" mass="{:.9g}" fullinertia="{:s}"'.format(
                    " ".join(f"{x:.9g}" for x in c1),
                    m1,
                    " ".join(f"{x:.9g}" for x in sym(i1)),
                ),
                lines[inertial_idx],
            )
            lines[inertial_idx] = new_line

    if not check and changed:
        open(path, "w").write("\n".join(lines))

    print(f"{os.path.basename(path):<42s} {len(body_spans(lines)):2d} body, "
          f"改 {changed:2d} 个   {before*1000:8.2f} → {after*1000:8.2f} g")
    return after - before


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--check", action="store_true", help="只打印，不写回")
    ap.add_argument("-v", "--verbose", action="store_true", help="逐 body 打印")
    args = ap.parse_args()

    files = sorted(
        f for f in glob.glob(os.path.join(MODEL_DIR, "robot_*.xml"))
        if os.path.basename(f).startswith(
            ("robot_walk", "robot_groundcontact", "robot_allcollisions")
        )
    )
    print("=" * 96)
    total = 0.0
    for f in files:
        total += process(f, args.check, args.verbose)
        print("-" * 96)
    print(f"合计增加 {total * 1000:.2f} g（每次运行的增量；--check 时也打印）")


if __name__ == "__main__":
    main()
