from fastapi import FastAPI
from router.licenseplate_detector_router import router

app = FastAPI()
app.include_router(router, prefix='/detector')


@app.get('/healthcheck', status_code=200)
async def healthcheck():
    return 'Licenseplate detector is ready!'
