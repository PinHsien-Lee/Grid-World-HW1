# HW1 — Grid World: Iterative Policy Evaluation & Value Iteration

## Summary
This project is a Flask-based interactive Grid World web application designed to visualize **Iterative Policy Evaluation** and **Value Iteration** in Reinforcement Learning. It features a side-by-side UI to compare a Random Deterministic Policy against an Optimal Policy found via Value Iteration. 

The application offers a dynamic, user-friendly interface to configure custom grid environments (selecting start, end, and obstacles) and adjust risk/reward parameters such as noise (slip probability), discount factor ($\gamma$), and step cost, with instant real-time recalculation of the state values $V(s)$.

---

## Project Structure

```
hw1/
├── app.py
├── requirements.txt
├── README.md
└── templates/
    └── index.html
```

---

## Features

### HW1-1: Grid Map Development
- Supports **n × n grid** where n is selectable from **5 to 9**
- Synchronized dual-board layout. Clicking on either board mirrors the setup:
  - **Start cell** — shown in green
  - **End / Goal cell** — shown in red
  - **n − 2 obstacle cells** — shown in gray
- Reset button regenerates a fresh grid of the same size instantly.

### HW1-2: Side-by-Side Policy Comparison
- **Left Board (Random Policy):** Each valid cell is randomly assigned **1 action** (up / down / left / right) as a **deterministic policy**. Iterative Policy Evaluation computes $V(s)$ for this specific policy. A "Re-Evaluate" button allows rolling a new random policy on the same board.
- **Right Board (Value Iteration):** Uses the Value Iteration algorithm to compute the mathematically **optimal policy** for the given grid and risk parameters. It dynamically displays the best actions to take from every state to maximize expected reward.
- Actions are displayed as **directional arrows** inside each cell, and the state value $V(s)$ is displayed below the arrows.
- Cells are dynamically colored via a heatmap reflecting their $V(s)$ value.

### HW1-3: Risk & Reward Parameter Tuning
Interactive sliders are provided to instantly recalculate and observe agent behavior changes under different risk profiles:
- **Noise/Slip Probability (Risk):** Simulates the chance of the agent sliding orthogonal to its intended direction (e.g., trying to go Up, but sliding Left/Right).
- **Discount Factor ($\gamma$):** Controls whether the agent prefers immediate rewards or distant goals.
- **Step Cost:** Controls the penalty applied for every step taken. A high step cost creates urgency to reach the goal faster.

*(Adjusting any slider instantly recalculates both the Random Policy evaluation and the Value Iteration optimal policy in real-time).*

---

## Installation & Usage

```bash
# 1. (Optional) Create a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the Flask server
python app.py
```

Then open your browser and go to: [http://127.0.0.1:5000](http://127.0.0.1:5000)

---

## How to Use

1. Select grid size **n** (5–9) from the dropdown.
2. Click **Generate Grid** to create the synchronized dual maps.
3. Click a cell on either board → sets the **Start** (green).
4. Click another cell → sets the **End / Goal** (red).
5. Click **n − 2** more cells → sets **Obstacles** (gray).
6. Click **Evaluate Policy** → calculates and displays the Random Policy (left) and the Optimal Value Iteration Policy (right).
7. Adjust the **Noise**, **Discount Factor ($\gamma$)**, or **Step Cost** sliders to watch how the Optimal Policy actively adapts to high/low risk scenarios instantly without re-clicking evaluate.
8. Click **Re-evaluate Policy** under the left board to test a new random policy.
9. Click **Reset** to clear everything and start over.
