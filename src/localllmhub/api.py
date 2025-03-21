from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import os
from importlib import resources

app = FastAPI()
with resources.path("localllmhub", "templates") as template_dir:
    print(template_dir)
    templates = Jinja2Templates(directory=str(template_dir))

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/chat", response_class=HTMLResponse)
async def chat(request: Request):
    return templates.TemplateResponse("chat.html", {"request": request})

@app.get("/llm-manager", response_class=HTMLResponse)
async def llm_manager(request: Request):
    return templates.TemplateResponse("llm-manager.html", {"request": request})

@app.get("/api/list-llms")
async def list_llms():
    cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
    
    if not os.path.exists(cache_dir):
        return {"error": f"Cache directory {cache_dir} not found"}
    
    models = [d for d in os.listdir(cache_dir) if os.path.isdir(os.path.join(cache_dir, d))]
    
    llm_info = []
    
    for model in models:
        model_path = os.path.join(cache_dir, model)
        total_size = 0
        for dirpath, _, filenames in os.walk(model_path):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                total_size += os.path.getsize(fp)
        
        size_mb = total_size / (1024 * 1024)
        model_name = model.replace("models--", "").replace("--", "/")
        llm_info.append({
            "name": model_name,
            "size_mb": round(size_mb, 2)
        })
    
    return {"llms": llm_info}