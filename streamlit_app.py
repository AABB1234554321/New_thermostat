import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# --- App Title and Description ---
st.title("Thermostat Simulation: Comparing Control Algorithms")
st.write("This interactive simulation compares On-Off, PID, and Q-Learning control algorithms for maintaining room temperature.")

# --- Input Parameters ---
st.sidebar.header("Simulation Parameters")
initial_room_temperature = st.sidebar.number_input("Initial Room Temperature (°C)", min_value=10, max_value=30, value=19)
outside_temperature = st.sidebar.number_input("Outside Temperature (°C)", min_value=0, max_value=40, value=10)
thermostat_setting = st.sidebar.number_input("Thermostat Setting (°C)", min_value=15, max_value=25, value=20)
heater_power = st.sidebar.slider("Heater Power (°C/minute)", min_value=0.1, max_value=0.5, value=0.3)
heat_loss = st.sidebar.slider("Heat Loss (°C/minute)", min_value=0.05, max_value=0.2, value=0.1)
simulation_minutes = st.sidebar.number_input("Simulation Minutes", min_value=10, max_value=120, value=60)
thermostat_sensitivity = st.sidebar.slider("Thermostat Sensitivity (°C)", min_value=0.1, max_value=0.5, value=0.5, step=0.1)

# --- Q-Learning Parameters ---
st.sidebar.subheader("Q-Learning Parameters")
episodes = st.sidebar.number_input("Training Episodes", min_value=100, max_value=5000, value=1000)
learning_rate = 0.1  # Fixed for simplicity
discount_factor = 0.95  # Fixed for simplicity
exploration_rate = 0.1  # Fixed for simplicity

# --- PID Parameters ---
st.sidebar.subheader("PID Parameters")
Kp = st.sidebar.slider("Kp (Proportional Gain)", min_value=0.1, max_value=2.0, value=0.5)
Ki = st.sidebar.slider("Ki (Integral Gain)", min_value=0.01, max_value=0.5, value=0.1)
Kd = st.sidebar.slider("Kd (Derivative Gain)", min_value=0.001, max_value=0.2, value=0.01)

# --- Global Variables ---
num_states = 41
num_actions = 2
q_table = np.zeros((num_states, num_actions))  # Initialize q_table here

# --- Helper Functions ---
def get_state(temperature):
    """Discretizes the temperature into states."""
    return int(min(40, max(0, (temperature - 10) / 0.5)))

def get_action(state, q_table, exploration_rate):
    """Chooses an action based on the epsilon-greedy policy."""
    if np.random.uniform(0, 1) < exploration_rate:
        return np.random.choice(num_actions)  # Exploration
    else:
        return np.argmax(q_table[state, :])  # Exploitation

def get_reward(state, action, thermostat_setting):
    """Calculates the reward based on the state and action."""
    state_temp = 10 + state * 0.5
    if abs(state_temp - thermostat_setting) <= 0.5:
        return 10  # Within acceptable range
    elif action == 1 and state_temp > thermostat_setting + 0.5:  # Too hot
        return -10
    elif action == 0 and state_temp < thermostat_setting - 0.5:  # Too cold
        return -5
    else:
        return -1  # Slight penalty for not being in range

# --- Simulation Logic (On-Off) ---
def run_on_off_simulation(initial_room_temperature, thermostat_sensitivity):
    time = []
    room_temperatures = []
    room_temperature = initial_room_temperature
    heater_status = False

    for minute in np.arange(0, simulation_minutes, 0.1):
        time.append(minute)

        if room_temperature < thermostat_setting - thermostat_sensitivity:
            heater_status = True
        elif room_temperature > thermostat_setting + thermostat_sensitivity:
            heater_status = False

        if heater_status:
            room_temperature += heater_power * 0.1
        else:
            room_temperature -= heat_loss * 0.1

        room_temperatures.append(room_temperature)

    area_on_off = calculate_area_between_temp(time, room_temperatures, thermostat_setting)
    return time, room_temperatures, area_on_off  # Return area_on_off

# --- Simulation Logic (Q-Learning) ---
def run_q_learning_simulation(initial_room_temperature, thermostat_sensitivity):
    global q_table  # Ensure we're using the global q_table
    for episode in range(episodes):
        room_temperature = initial_room_temperature
        state = get_state(room_temperature)
        for _ in np.arange(0, simulation_minutes, 0.1):
            action = get_action(state, q_table, exploration_rate)
            if action == 1:
                room_temperature += heater_power * 0.1
            else:
                room_temperature -= heat_loss * 0.1

            next_state = get_state(room_temperature)
            reward = get_reward(next_state, action, thermostat_setting)

            q_table[state, action] += learning_rate * (reward + discount_factor * np.max(q_table[next_state, :]) - q_table[state, action])
            state = next_state

    # Run one final simulation using the learned Q-table
    time = []
    room_temperatures = []

    room_temperature = initial_room_temperature
    state = get_state(room_temperature)
    for minute in np.arange(0, simulation_minutes, 0.1):
        action = np.argmax(q_table[state, :])  # Always choose the best action

        if action == 1:
            room_temperature += heater_power * 0.1
        else:
            room_temperature -= heat_loss * 0.1

        state = get_state(room_temperature)
        time.append(minute)
        room_temperatures.append(room_temperature)

    area_q = calculate_area_between_temp(time, room_temperatures, thermostat_setting)
    return time, room_temperatures, area_q  # Return area_q

# --- Simulation Logic (PID) ---
def run_pid_simulation(initial_room_temperature, thermostat_sensitivity):
    time = []
    room_temperatures = []
    heater_output = []

    integral_error = 0
    previous_error = 0
    room_temperature = initial_room_temperature

    for minute in np.arange(0, simulation_minutes, 0.1):
        time.append(minute)

        error = thermostat_setting - room_temperature
        proportional_term = Kp * error
        integral_error += error * 0.1
        integral_term = Ki * integral_error
        derivative_term = Kd * (error - previous_error) / 0.1
        previous_error = error

        pid_output = proportional_term + integral_term + derivative_term
        heater_output_percent = max(0, min(1, pid_output))
        heater_output.append(heater_output_percent)

        room_temperature += (heater_power * heater_output_percent - heat_loss) * 0.1
        room_temperatures.append(room_temperature)

    area_pid = calculate_area_between_temp(time, room_temperatures, thermostat_setting)
    return time, room_temperatures, area_pid  # Return area_pid

# --- Calculate Area Between Current Temperature and Set Temperature ---

def calculate_area_between_temp(time, room_temperatures, set_temp):
    area = 0
    for i in range(1, len(time)):
        dt = time[i] - time[i - 1]
        area += abs(room_temperatures[i] - set_temp) * dt  # average_temp kullanımı kaldırıldı
    return area
def calculate_area_metrics(time, room_temperatures, set_temp):
    overshoot = 0
    undershoot = 0
    for i in range(1, len(time)):
        dt = time[i] - time[i - 1]
        avg_temp = (room_temperatures[i] + room_temperatures[i - 1]) / 2
        if avg_temp > set_temp:
            overshoot += (avg_temp - set_temp) * dt
        elif avg_temp < set_temp:
            undershoot += (set_temp - avg_temp) * dt
    total_area = overshoot + undershoot
    return overshoot, undershoot, total_area
# --- Main App ---
simulation_type = st.sidebar.multiselect("Choose Simulation Type(s):", ["On-Off", "Q-Learning", "PID"])

if st.sidebar.button("Run Simulation"):
    results = {}

    if "On-Off" in simulation_type:
        time_on_off, room_temperatures_on_off, area_on_off = run_on_off_simulation(initial_room_temperature, thermostat_sensitivity)
        st.write(f"**On-Off Control:** Area between current temperature and set temperature: {area_on_off:.2f} °C*minutes")
        results["On-Off"] = {'time': time_on_off, 'room_temperatures': room_temperatures_on_off}

    if "Q-Learning" in simulation_type:
        time_q, room_temperatures_q, area_q = run_q_learning_simulation(initial_room_temperature, thermostat_sensitivity)
        st.write(f"**Q-Learning:** Area between current temperature and set temperature: {area_q:.2f} °C*minutes")
        results["Q-Learning"] = {'time': time_q, 'room_temperatures': room_temperatures_q}

    if "PID" in simulation_type:
        time_pid, room_temperatures_pid, area_pid = run_pid_simulation(initial_room_temperature, thermostat_sensitivity)
        st.write(f"**PID Control:** Area between current temperature and set temperature: {area_pid:.2f} °C*minutes")
        results["PID"] = {'time': time_pid, 'room_temperatures': room_temperatures_pid}
# --- Plotting Results ---
    fig1, ax1 = plt.subplots(figsize=(12, 6))

    for algo, data in results.items():
        ax1.plot(data['time'], data['room_temperatures'], label=f"Room Temperature ({algo})")

    ax1.axhline(y=thermostat_setting, color='r', linestyle='--', label="Thermostat Setting")
    ax1.set_xlabel("Time (Minutes)")
    ax1.set_ylabel("Temperature (°C)")
    ax1.legend()
    ax1.grid(True)
    ax1.set_title("Room Temperature Control Simulation")

    st.pyplot(fig1)  # Ana grafiği göster

    # Bar Chart for Comfort and Energy Metrics
    fig2, ax2 = plt.subplots(figsize=(10, 4))  # Bar grafiği için yeni bir figure oluştur
    metrics = {algo: calculate_area_metrics(data['time'], data['room_temperatures'], thermostat_setting) for algo, data in results.items()}
    labels = list(metrics.keys())
    overshoot_values = [m[0] for m in metrics.values()]
    undershoot_values = [m[1] for m in metrics.values()]
    total_values = [m[2] for m in metrics.values()]

    width = 0.2
    x = np.arange(len(labels))

    ax2.bar(x - width, overshoot_values, width, label='Overshoot', color='skyblue')
    ax2.bar(x, undershoot_values, width, label='Undershoot', color='lightcoral')
    ax2.bar(x + width, total_values, width, label='Total', color='lightgreen')

    ax2.set_ylabel('Area (°C*minutes)')
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels)
    ax2.legend(loc='upper right')
    ax2.set_title("Comfort and Energy Consumption Metrics")

    st.pyplot(fig2)  # Bar grafiğini göster

