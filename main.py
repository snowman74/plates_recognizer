from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from routers import monolith, api


app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
app.router.include_router(monolith.router)
app.router.include_router(api.router)
