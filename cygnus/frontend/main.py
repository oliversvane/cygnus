from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import json
import random
import asyncio

app = FastAPI()


# WebSocket endpoint
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    
    # Simulating sending data (e.g., anomalies detected) at regular intervals
    try:
        while True:
            data = {"x": random.uniform(0, 10), "y": random.uniform(0, 10), "label": random.choice([0, 1])}
            await websocket.send_text(json.dumps(data))
            await asyncio.sleep(1)  # Send data every second
    except WebSocketDisconnect:
        print("Client disconnected")

# Home page
@app.get("/")
async def read_root():
    return {"message": "Welcome to the anomaly detection WebSocket example!"}
