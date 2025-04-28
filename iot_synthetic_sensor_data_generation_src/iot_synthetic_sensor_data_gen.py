import pandas as pd
import random
import os

def generate_synthetic_data(n_samples=10000):
    data = {
        "temperature": [],        # DHT22 (°C)
        "humidity": [],           # DHT22 (%)
        "mq2_smoke": [],          # MQ-2 (analog: 0–1023)
        "mq135_gas": [],          # MQ-135 (analog: 0–1023)
        "flame_detected": [],     # Flame sensor (0/1)
        "cv_flame_score": [],     # Vision model (0.0–1.0)
        "cv_smoke_score": [],     # Vision model (0.0–1.0)
        "person_detected": [],    # Vision model (0/1)
        "label": []
    }

    for _ in range(n_samples):
        scenario = random.random()

        if scenario < 0.3:
            # Safe
            data["temperature"].append(round(random.uniform(20, 35), 2))
            data["humidity"].append(round(random.uniform(30, 60), 2))
            data["mq2_smoke"].append(random.randint(100, 200))
            data["mq135_gas"].append(random.randint(100, 250))
            data["flame_detected"].append(0)
            data["cv_flame_score"].append(round(random.uniform(0, 0.1), 2))
            data["cv_smoke_score"].append(round(random.uniform(0, 0.1), 2))
            data["person_detected"].append(random.randint(0, 1))
            data["label"].append("Safe")

        elif scenario < 0.5:
            # Warning
            data["temperature"].append(round(random.uniform(35, 45), 2))
            data["humidity"].append(round(random.uniform(30, 60), 2))
            data["mq2_smoke"].append(random.randint(200, 400))
            data["mq135_gas"].append(random.randint(200, 400))
            data["flame_detected"].append(0)
            data["cv_flame_score"].append(round(random.uniform(0, 0.3), 2))
            data["cv_smoke_score"].append(round(random.uniform(0.2, 0.5), 2))
            data["person_detected"].append(random.randint(0, 1))
            data["label"].append("Warning")

        elif scenario < 0.7:
            # Gas Leak
            data["temperature"].append(round(random.uniform(35, 50), 2))
            data["humidity"].append(round(random.uniform(20, 50), 2))
            data["mq2_smoke"].append(random.randint(300, 600))
            data["mq135_gas"].append(random.randint(600, 900))
            data["flame_detected"].append(0)
            data["cv_flame_score"].append(round(random.uniform(0.1, 0.4), 2))
            data["cv_smoke_score"].append(round(random.uniform(0.3, 0.6), 2))
            data["person_detected"].append(random.randint(0, 1))
            data["label"].append("Gas Leak")

        elif scenario < 0.9:
            # Fire Detected
            data["temperature"].append(round(random.uniform(45, 70), 2))
            data["humidity"].append(round(random.uniform(10, 40), 2))
            data["mq2_smoke"].append(random.randint(700, 900))
            data["mq135_gas"].append(random.randint(600, 850))
            data["flame_detected"].append(1)
            data["cv_flame_score"].append(round(random.uniform(0.6, 0.9), 2))
            data["cv_smoke_score"].append(round(random.uniform(0.6, 0.9), 2))
            data["person_detected"].append(random.randint(0, 1))
            data["label"].append("Fire Detected")

        else:
            # Evacuate Immediately
            data["temperature"].append(round(random.uniform(65, 100), 2))
            data["humidity"].append(round(random.uniform(5, 30), 2))
            data["mq2_smoke"].append(random.randint(850, 1023))
            data["mq135_gas"].append(random.randint(850, 1023))
            data["flame_detected"].append(1)
            data["cv_flame_score"].append(round(random.uniform(0.9, 1.0), 2))
            data["cv_smoke_score"].append(round(random.uniform(0.9, 1.0), 2))
            data["person_detected"].append(0)
            data["label"].append("Evacuate Immediately")

    df = pd.DataFrame(data)
    cwd = os.getcwd()
    os.makedirs("data",exist_ok=True)
    data_path = os.path.join(cwd,"data","iot_synthetic_sensor_data.csv")
    df.to_csv(data_path, index=False)
    print(f"✅ Generated {n_samples} samples and saved as 'iot_synthetic_sensor_data.csv'")

if __name__ == "__main__":
    generate_synthetic_data(1000)
