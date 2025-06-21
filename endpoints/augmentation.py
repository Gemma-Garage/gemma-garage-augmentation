from fastapi import APIRouter

router = APIRouter()

@router.get("/parse")
async def select_model(model_name: str):
    # Here you might check if the model_name is supported, load metadata, etc.
    return {"message": f"Model '{model_name}' selected"}