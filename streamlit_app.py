import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# --- App Title and Description ---
st.title("Thermostat Simulation with Q-Learning and PID Control")
st.write("This simulation compares Q-Learning and PID control for maintaining room temperature.")

# --- Input Parameters ---
initial_room_temperature = st.number_input("Initial Room Temperature (°C)", min_value=10, max_value=30, value=19)
outside_temperature = st.number_input("Outside Temperature (°C)", min_value=0, max_value=40, value=10)
thermostat_setting = st.number_input("Thermostat Setting (°C)", min_value=15, max_value=25, value=20)
heater_power = st.slider("Heater Power (°C/minute)", min_value=0.1, max_value=0.5, value=0.3)
heat_loss = st.slider("Heat Loss (°C/minute)", min_value=0.05, max_value=0.2, value=0.1)

# Q-learning Parameters
num_states = 41
num_actions = 2
q_table = np.zeros((num_states, num_actions))
learning_rate = 0.1
discount_factor = 0.95
exploration_rate = 0.1
episodes = st.number_input("Training Episodes (Q-Learning)", min_value=100, max_value=5000, value=1000)

# Simulation Parameters
simulation_minutes = st.number_input("Simulation Minutes", min_value=10, max_value=120, value=60)

# PID Parameters
Kp = st.slider("Kp (Proportional Gain)", min_value=0.1, max_value=2.0, value=0.5)
Ki = st.slider("Ki (Integral Gain)", min_value=0.01, max_value=0.5, value=0.1)
Kd = st.slider("Kd (Derivative Gain)", min_value=0.001, max_value=0.2, value=0.01)

# --- Helper Functions (Q-Learning) ---
def get_state(temperature):
    return int((temperature - 10) // 0.5)

def get_action(state):
    if np.random.uniform(0, 1) < exploration_rate:
        return np.random.choice(num_actions)  # Exploration
    else:
        return np.argmax(q_table[state, :])   # Exploitation

def get_reward(state, action, thermostat_setting):
    state_temp = 10 + state * 0.5
    if abs(state_temp - thermostat_setting) <= 0.5:
        return 10  # Within acceptable range
    elif action == 1 and state_temp > thermostat_setting + 0.5:  # Too hot
        return -10
    elif action == 0 and state_temp < thermostat_setting - 0.5:  # Too cold
        return -5
    else:
        return -1  # Slight penalty for not being in range

def run_q_learning_simulation(initial_room_temperature):
    global q_table  # Ensure we're using the global q_table
    for episode in range(episodes):
        room_temperature = initial_room_temperature
        state = get_state(room_temperature)
        for _ in np.arange(0, simulation_minutes, 0.1):
            action = get_action(state)
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
    heater_output = []

    room_temperature = initial_room_temperature
    state = get_state(room_temperature)
    for minute in np.arange(0, simulation_minutes, 0.1):
        action = np.argmax(q_table[state, :])  # Always choose the best action
        heater_output.append(action)

        if action == 1:
            room_temperature += heater_power * 0.1
        else:
            room_temperature -= heat_loss * 0.1

        state = get_state(room_temperature)
        time.append(minute)
        room_temperatures.append(room_temperature)

    df = pd.DataFrame({
        'Time (Minutes)': time,
        'Room Temperature (°C)': room_temperatures,
        'Heater Output': heater_output
    })

    return time, room_temperatures, heater_output, df

# --- Simulation Logic (PID) ---
def run_pid_simulation(initial_room_temperature):
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

    df = pd.DataFrame({
        'Time (Minutes)': time,
        'Room Temperature (°C)': room_temperatures,
        'Heater Output (%)': heater_output
    })

    return time, room_temperatures, heater_output, df

# --- Calculate Area Between Current Temperature and Set Temperature ---
def calculate_area_between_temp(time, room_temperatures, set_temp):
    area = 0
    for i in range(1, len(time)):
        dt = time[i] - time[i - 1]
        avg_temp = (room_temperatures[i] + room_temperatures[i - 1]) / 2
        area += abs(avg_temp - set_temp) * dt
    return area

# --- Main App ---
simulation_type = st.selectbox("Choose Simulation Type:", ("Q-Learning", "PID", "Both"))

if st.button("Run Simulation"):
    time_q = room_temperatures_q = heater_output_q = df_q = None
    time_pid = room_temperatures_pid = heater_output_pid = df_pid = None

    if simulation_type == "Q-Learning" or simulation_type == "Both":
        time_q, room_temperatures_q, heater_output_q, df_q = run_q_learning_simulation(initial_room_temperature)
        area_q = calculate_area_between_temp(time_q, room_temperatures_q, thermostat_setting)
        st.write(f"Q-Learning Area Between Current Temp and Set Temp: {area_q:.2f} °C*minutes")

    if simulation_type == "PID" or simulation_type == "Both":
        time_pid, room_temperatures_pid, heater_output_pid, df_pid = run_pid_simulation(initial_room_temperature)
        area_pid = calculate_area_between_temp(time_pid, room_temperatures_pid, thermostat_setting)
        st.write(f"PID Area Between Current Temp and Set Temp: {area_pid:.2f} °C*minutes")

    # --- Plotting Results ---
    plt.figure(figsize=(12, 6))

    if simulation_type == "Q-Learning" and time_q is not None:
        plt.plot(time_q, room_temperatures_q, label="Room Temperature (Q-Learning)", color="blue")
        plt.plot(time_q, heater_output_q, label="Heater Output (Q-Learning)", color="lightblue", linestyle="--")
    if simulation_type == "PID" and time_pid is not None:
        plt.plot(time_pid, room_temperatures_pid, label="Room Temperature (PID)", color="orange")
        plt.plot(time_pid, heater_output_pid, label="Heater Output (PID)", color="coral", linestyle="--")
    if simulation_type == "Both" and time_q is not None and time_pid is not None:
        plt.plot(time_q, room_temperatures_q, label="Room Temperature (Q-Learning)", color="blue")
        plt.plot(time_pid, room_temperatures_pid, label="Room Temperature (PID)", color="orange")
        plt.plot(time_q, heater_output_q, label="Heater Output (Q-Learning)", color="lightblue", linestyle="--")
        plt.plot(time_pid, heater_output_pid, label="Heater Output (PID)", color="coral", linestyle="--")

    # Adjust y-axis limits based on available data
    min_temp = min(room_temperatures_q) if room_temperatures_q is not None else (min(room_temperatures_pid) if room_temperatures_pid is not None else 19)
    max_temp = max(room_temperatures_q) if room_temperatures_q is not None else (max(room_temperatures_pid) if room_temperatures_pid is not None else 21)
    plt.axhline(y=thermostat_setting, color='r', linestyle='--', label="Thermostat Setting")
    plt.axhline(y=thermostat_setting + 0.5, color='g', linestyle='--', alpha=0.3, label="Acceptable Range")
    plt.axhline(y=thermostat_setting - 0.5, color='g', linestyle='--', alpha=0.3)
    plt.ylim(min_temp - 1, max_temp + 1)

    plt.xlabel("Time (Minutes)")
    plt.ylabel("Temperature (°C)")
    plt.legend()
    plt.grid(True)
    plt.title("Room Temperature Control")
    st.pyplot(plt)
