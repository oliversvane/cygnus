import threading
import paho.mqtt.client as mqtt

# MQTT Config
MQTT_BROKER = 'localhost'
MQTT_PORT = 1883
MQTT_TOPIC = 'my/topic'

# Dependency Example
def get_settings():
    return {"app_name": "FastAPI MQTT Boilerplate"}



# MQTT Handlers
def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print("Connected to MQTT Broker!")
        client.subscribe(MQTT_TOPIC)
    else:
        print(f"Failed to connect, return code {rc}")

def on_message(client, userdata, msg):
    print(f"Received message on {msg.topic}: {msg.payload.decode()}")

def start_mqtt():
    client = mqtt.Client()
    client.on_connect = on_connect
    client.on_message = on_message
    client.connect(MQTT_BROKER, MQTT_PORT, 60)
    client.loop_forever()

# Run MQTT in a thread
threading.Thread(target=start_mqtt, daemon=True).start()

