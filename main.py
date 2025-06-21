from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from endpoints import data_parser, augmentation

app = FastAPI(title="LLM Garage Data Augmentation API")

origins = [
    "http://localhost:3000",  # your React app's origin
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(data_parser.router, prefix="/parsing", tags=["Document Parsing"])
app.include_router(augmentation.router, prefix="/augment", tags=["Data Augmentation"])


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
