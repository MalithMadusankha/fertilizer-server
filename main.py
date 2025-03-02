from fastapi import FastAPI
# from routes.predict import predict

from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

previous_val = "None"


# Set up CORS
origins = [
    "http://localhost",
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# app.include_router(predict)


@app.get("/")
async def read_root():
    print("<========= Calll Default route ==========>")
    return {"message": "Hello !!!, I am FastAPI Server. U can call my API I am here to responed "}

print("<============== Server started ==============>")
