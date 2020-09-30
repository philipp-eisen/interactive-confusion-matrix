import uuid
from tempfile import TemporaryFile

from fastapi import FastAPI
from starlette.responses import HTMLResponse, PlainTextResponse, FileResponse

import altair as alt
from icm.models import ConfusionMatrixRequest
from icm.plot import cm_chart

app = FastAPI(title="Interactive Confusion Matrix")


@app.get(
    "/",
    summary="Status of the service",
    description="Status of service, response code HTTP 200 - OK when up",
    response_class=PlainTextResponse,
)
def status():
    return


@app.post("/matrix/download")
def cm(req: ConfusionMatrixRequest):
    plot: alt.Chart = cm_chart(req)
    with open("confusion-matrix.html", "w") as f:
        plot.save(f.name, format="html")
    return FileResponse("confusion-matrix.html", filename="confusion-matrix.html")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8080)
