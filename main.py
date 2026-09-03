import os
import datetime as dt
import time
import uvicorn
import time

if __name__ == "__main__":
    version = '1.1.2'
    print(f"Start version {version}")
    uvicorn.run("app.main:app", host="0.0.0.0",
                port=8080, proxy_headers=True)
