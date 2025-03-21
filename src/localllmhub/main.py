import uvicorn

def main():
    uvicorn.run("localllmhub.api:app", host="127.0.0.1", port=6969, log_level="info", reload=True)

if __name__ == "__main__":
    main()