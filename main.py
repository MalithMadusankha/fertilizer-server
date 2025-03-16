import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routes.fertilizer import fertilizer
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse

app = FastAPI()

# Dynamically get the absolute path of the 'results' directory
RESULTS_DIR = os.path.abspath("results")
print(RESULTS_DIR)
# Ensure the directory exists
if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)

print("Results Directory Absolute Path:", RESULTS_DIR)
print("Contents:", os.listdir(RESULTS_DIR) if os.path.exists(RESULTS_DIR) else "Not Found")

# CORS Setup
origins = ["http://localhost", "http://localhost:3000", "https://*.railway.app"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/results", StaticFiles(directory=RESULTS_DIR, html=True), name="results")


@app.get("/")
async def read_root():
    print("<========= Call Default route ==========>")
    return {"message": "Hello !!!, I am FastAPI Server. U can call my API I am here to respond"}

@app.get("/list-results")
async def list_results():
    files = os.listdir(RESULTS_DIR)
    return JSONResponse(content={"files": files})

app.include_router(fertilizer)

print("<============== Server started ==============>")

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
