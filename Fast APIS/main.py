from fastapi import FastAPI
app = FastAPI()
@app.get("/")
def read_root():
  return "Try the new"
@app.post("/items/")
def create_item(item: dict):
  return item