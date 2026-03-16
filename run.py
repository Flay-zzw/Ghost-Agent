# run.py
import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "main:app",  # 如果你的文件在子文件夹中，使用 "app.main:app"
        host="localhost",
        port=8000,
        reload=True,
        log_level="info"
    )