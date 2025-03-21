from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import os
from importlib import resources
from huggingface_hub import hf_hub_download, list_repo_files, get_hf_file_metadata, hf_hub_url
from starlette.responses import StreamingResponse
from fastapi import HTTPException
import logging
import asyncio
from transformers import pipeline
import torch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()
with resources.path("localllmhub", "templates") as template_dir:
    templates = Jinja2Templates(directory=str(template_dir))

loaded_pipelines = {}

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    hf_token = os.getenv('HF_TOKEN', '')
    return templates.TemplateResponse("index.html", {
        "request": request,
        "hf_token": hf_token
    })

@app.post("/api/set-token")
async def set_token(request: Request):
    data = await request.json()
    if not data or 'hf_token' not in data:
        raise HTTPException(status_code=400, detail="No token provided")
    
    hf_token = data['hf_token']
    os.environ['HF_TOKEN'] = hf_token
    return {"message": "Token updated successfully"}

@app.get("/chat", response_class=HTMLResponse)
async def chat(request: Request):
    return templates.TemplateResponse("chat.html", {"request": request})

@app.get("/llm-manager", response_class=HTMLResponse)
async def llm_manager(request: Request):
    return templates.TemplateResponse("llm-manager.html", {"request": request})

@app.get("/downloader", response_class=HTMLResponse)
async def downloader(request: Request):
    return templates.TemplateResponse("downloader.html", {"request": request})

async def download_progress(repo_id: str):
    try:
        files = list_repo_files(repo_id=repo_id)
        essential_files = [f for f in files if f.endswith(('.safetensors', '.json')) and not f.startswith(('pytorch_', 'flax_'))]
        total_size = 0
        downloaded_size = 0
        
        for file in essential_files:
            metadata = get_hf_file_metadata(hf_hub_url(repo_id, file))
            total_size += metadata.size

        yield f"data: {{\"status\": \"initializing\", \"message\": \"Found {len(essential_files)} essential files, total size: {(total_size / (1024 * 1024)):.2f} MB\"}}\n\n"

        for file in essential_files:
            yield f"data: {{\"status\": \"downloading\", \"file\": \"{file}\", \"progress\": {(downloaded_size / total_size * 100):.1f}\"}}\n\n"
            
            file_path = hf_hub_download(
                repo_id=repo_id,
                filename=file,
                token=os.getenv('HF_TOKEN', None)
            )
            
            file_size = os.path.getsize(file_path)
            downloaded_size += file_size
            
            percentage = (downloaded_size / total_size) * 100
            yield f"data: {{\"status\": \"downloading\", \"file\": \"{file}\", \"progress\": {percentage:.1f}}}\n\n"

        yield "data: {\"status\": \"completed\", \"progress\": 100.0}\n\n"
        
    except Exception as e:
        logger.error(f"Error downloading model {repo_id}: {str(e)}")
        yield f"data: {{\"status\": \"error\", \"message\": \"{str(e)}\"}}\n\n"

@app.get("/api/install-llm")
async def install_llm(name: str):
    if not name or "/" not in name:
        raise HTTPException(status_code=400, detail="Invalid model name.")
    logger.info(f"Starting installation of model: {name}")
    return StreamingResponse(
        download_progress(name),
        media_type="text/event-stream"
    )

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
        for dirpath, dirname, filenames in os.walk(model_path):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                total_size += os.path.getsize(fp)
        
        size_mb = total_size / (1024 * 1024)
        model_name = model.replace("models--", "").replace("--", "/")
        if model_name.startswith("."):
            continue
        llm_info.append({
            "name": model_name,
            "size_mb": round(size_mb, 2)
        })
    
    return {"llms": llm_info}

@app.post("/api/chat")
async def chat_with_llm(request: Request):
    data = await request.json()
    model_name = data.get("model")
    message = data.get("message")

    if not model_name or not message:
        raise HTTPException(status_code=400, detail="Model name and message are required")

    if model_name not in loaded_pipelines:
        try:
            loaded_pipelines[model_name] = pipeline(
                "text-generation",
                model=model_name,
                torch_dtype=torch.bfloat16,
                device=0 if torch.cuda.is_available() else -1
            )
        except Exception as e:
            logger.error(f"Error loading model {model_name}: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")

    pipe = loaded_pipelines[model_name]

    messages = [
        {"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant."}]},
        {"role": "user", "content": [{"type": "text", "text": message}]}
    ]

    output = pipe(messages, max_new_tokens=100)
    response = output[0]["generated_text"][-1]["content"]

    return {"response": response}