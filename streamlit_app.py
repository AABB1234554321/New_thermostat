import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from geneticalgorithm import geneticalgorithm as ga
# --- App Title and Description ---
st.title("Thermostat Simulation with Q-Learning, PID, and ON-OFF Control")
st.write("This simulation compares Q-Learning, PID control, and ON-OFF control for maintaining room temperature.")

# --- Input Parameters ---
initial_room_temperature = st.number_input("Initial Room Temperature (°C)", min_value=10, max_value=30, value=19)
outside_temperature = st.number_input("Outside Temperature (°C)", min_value=0, max_value=40, value=10)
thermostat_setting = st.number_input("Thermostat Setting (°C)", min_value=15, max_value=25, value=20)
heater_power = st.slider("Heater Power (°C/minute)", min_value=0.1, max_value=0.5, value=0.3)
base_heat_loss = st.slider("Base Heat Loss (°C/minute)", min_value=0.05, max_value=0.2, value=0.1)

# Q-learning Parameters
num_states = 41
num_actions = 2
q_table = np.zeros((num_states, num_actions))
learning_rate = 0.1
discount_factor = 0.9
exploration_rate = 0.1
episodes = st.number_input("Training Episodes (Q-Learning)", min_value=100, max_value=5000, value=1000)

# Simulation Parameters
simulation_minutes = st.number_input("Simulation Minutes", min_value=10, max_value=1440, value=60)

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
                heat_loss = base_heat_loss * (room_temperature - outside_temperature)
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
            heat_loss = base_heat_loss * (room_temperature - outside_temperature)
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
def run_pid_simulation(initial_room_temperature, Kp, Ki, Kd):
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

        heat_loss = base_heat_loss * (room_temperature - outside_temperature)
        room_temperature += (heater_power * heater_output_percent - heat_loss) * 0.1
        room_temperatures.append(room_temperature)

    df = pd.DataFrame({
        'Time (Minutes)': time,
        'Room Temperature (°C)': room_temperatures,
        'Heater Output (%)': heater_output
    })

    return time, room_temperatures, heater_output, df

# --- Simulation Logic (ON-OFF) ---
def run_on_off_simulation(initial_room_temperature):
    time = []
    room_temperatures = []
    heater_output = []

    room_temperature = initial_room_temperature
    for minute in np.arange(0, 60, 0.1):
        time.append(minute)  # Record time in minutes

        # Thermostat control
        if room_temperature < thermostat_setting - 0.5:
            heater_status = True  # Turn on the heater if below lower threshold
        elif room_temperature > thermostat_setting + 0.5:
            heater_status = False  # Turn off the heater if above upper threshold

        # Update room temperature based on heater status
        if heater_status:
            room_temperature += heater_power * 0.1  # Increase temperature if heater is on
        else:
            heat_loss = base_heat_loss * (room_temperature - outside_temperature)
            room_temperature -= heat_loss * 0.1  # Decrease temperature if heater is off

        heater_output.append(1 if heater_status else 0)
        room_temperatures.append(room_temperature)

    df = pd.DataFrame({
        'Time (Minutes)': time,
        'Room Temperature (°C)': room_temperatures,
        'Heater Output': heater_output
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

# --- Optimization Function for PID Parameters ---
def optimize_pid(params):
    Kp, Ki, Kd = params
    _, room_temperatures, _, _ = run_pid_simulation(initial_room_temperature, Kp, Ki, Kd)
    area = calculate_area_between_temp(np.arange(0, simulation_minutes, 0.1), room_temperatures, thermostat_setting)
    return area

# --- Main Streamlit App ---
def main():
    # ... (App title, description, input parameters remain the same)

    simulation_type = st.selectbox("Choose Simulation Type:", ("Q-Learning", "PID", "ON-OFF", "All"))

    if st.button("Run Simulation"):
        with st.spinner("Running simulation..."):
            results = {}
            sim_types = [simulation_type] if simulation_type != "All" else ["Q-Learning", "PID", "ON-OFF"]
            for sim_type in sim_types:
                try:
                    if sim_type == "Q-Learning":
                        results[sim_type] = run_q_learning_simulation(initial_room_temperature)
                    elif sim_type == "PID":
                        # Optimize PID parameters using Genetic Algorithm
                        varbound = np.array([[0.1, 1000.0], [0.00001, 0.5], [0.001, 0.9]])
                        algorithm_param = {
                            r'max_num_iteration': 100,
                            r'population_size': 50,
                            r'mutation_probability': 0.1,
                            r'elit_ratio': 0.01,
                            r'crossover_probability': 0.5,
                            r'parents_portion': 0.3,
                            r'crossover_type': r'uniform',
                            r'max_iteration_without_improv': None
                        }

                        model = ga(function=optimize_pid, dimension=3, variable_type='real', variable_boundaries=varbound, algorithm_parameters=algorithm_param)
                        model.run()
                        best_params = model.output_dict['variable']
                        st.write(f"Optimized PID Parameters: Kp={best_params[0]:.2f}, Ki={best_params[1]:.5f}, Kd={best_params[2]:.3f}")
                        results[sim_type] = run_pid_simulation(initial_room_temperature, *best_params)
                    elif sim_type == "ON-OFF":
                        results[sim_type] = run_on_off_simulation(initial_room_temperature)

                    time, room_temperatures, _, df = results[sim_type]
                    area = calculate_area_between_temp(time, room_temperatures, thermostat_setting)
                    st.write(f"Heat loss with {sim_type}: {area:.2f} °C*minutes")
                    # Plot Results
                    plt.figure(figsize=(12, 6))
                    plt.plot(time, room_temperatures, label=f"Oda Sıcaklığı ({sim_type})")
                    plt.plot(time, df['Heater Output'], label=f"Isıtıcı Çıktısı ({sim_type})", linestyle="--")

                except Exception as e:
                    st.error(f"Simülasyon hatası ({sim_type}): {e}")

        # Plot Settings (Common for all simulations)
        plt.axhline(y=thermostat_setting, color='r', linestyle='--', label="Termostat Ayarı")
        plt.axhline(y=thermostat_setting + 0.5, color='g', linestyle='--', alpha=0.3, label="Kabul Edilebilir Aralık")
        plt.axhline(y=thermostat_setting - 0.5, color='g', linestyle='--', alpha=0.3)
        plt.xlabel("Zaman (Dakika)")
        plt.ylabel("Sıcaklık (°C)")
        plt.legend()
        plt.grid(True)
        plt.title("Oda Sıcaklığı Kontrolü")
        st.pyplot(plt)

# Run the app
if __name__ == "__main__":
    main()
