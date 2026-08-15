import random
import time
from app.storage import insert_readings

def generate_random_reading(well_id: str) -> dict:
    """Generates a random telemetry reading for a given well."""
    return {
        "timestamp": time.time(),
        "well_id": well_id,
        "var1": random.uniform(0, 100),
        "var2": random.uniform(0, 100),
        "var3": random.uniform(0, 100),
        "var4": random.uniform(0, 100),
        "var5": random.uniform(0, 100),
        "var6": random.uniform(0, 100),
        "var7": random.uniform(0, 100),
        "var8": random.uniform(0, 100),
        "var9": random.uniform(0, 100),
        "var10": random.uniform(0, 100),
    }

def main():
    well_ids = [f"Well {i+1}" for i in range(4)]  # Simulate 4 wells
    while True:
        for well_id in well_ids:
            reading = generate_random_reading(well_id)
            insert_readings(reading)
        time.sleep(1)  # Simulate a 1-second interval between readings

if __name__ == "__main__":
    main()