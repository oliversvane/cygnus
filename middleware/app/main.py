from fastapi import FastAPI
import uvicorn

app = FastAPI(title="Middleware")


# Route
@app.get("/", tags=["Root"])
def read_root():
    return {"message": "Hello World"}


# Run with: uvicorn main:app --reload
if __name__ == '__main__':
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
