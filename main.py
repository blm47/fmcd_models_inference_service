"""Запуск одного процесса API и GPU worker."""

import uvicorn

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8080, proxy_headers=True, workers=1)
